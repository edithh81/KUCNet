"""Measure subgraph size and PoS(L) for a trained KUCNet checkpoint.

Reproduces the efficiency numbers in the thesis section
"Subgraph size after sampling" (Figure 6.1 + Table 6.4):

  - Average number of nodes per layer in the final subgraph.
  - PI(L): average # ground-truth positive items retained in the final subgraph.
  - SI(L): average # items in the final subgraph.
  - PoS(L) = PI(L) / SI(L).

Usage:
  python measure_subgraph.py --data_path data/last-fm/ \
      --ckpt results/checkpoints/last-fm_best.pt --mode model
  python measure_subgraph.py --data_path data/last-fm/ \
      --ckpt results/checkpoints/last-fm_best.pt --mode none --max_users 200

Modes:
  model  - run with the loaded model's built-in sampling (whatever it was trained with).
           Use this for the "KUCNet" or "Our method" column, depending on the
           checkpoint you load.
  none   - bypass PPR top-K pruning in middle layers (no sampling). Memory-heavy:
           use --max_users and a small --batch_size.
"""
import argparse
import json
import os
import time
from types import MethodType

import numpy as np
import torch
import torch.nn as nn

from scatter_shim import scatter
from load_data import DataLoader
from models import KUCNet_trans


# Mirrors the per-dataset hyperparams in train.py. Keep in sync.
DATASET_CONFIGS = {
    'new_alibaba-fashion': dict(lr=0.00005, decay_rate=0.999, lamb=0.0001,
                                hidden_dim=48, attn_dim=5, n_layer=5,
                                dropout=0.01, act='idd', n_batch=20,
                                n_tbatch=20, K=50),
    'alibaba-fashion': dict(lr=10**-6.5, decay_rate=0.998, lamb=0.00001,
                            hidden_dim=48, attn_dim=5, n_layer=5,
                            dropout=0.2, act='relu', n_batch=10,
                            n_tbatch=10, K=70),
    'last-fm': dict(lr=0.0004, decay_rate=0.994, lamb=0.00014,
                    hidden_dim=48, attn_dim=5, n_layer=3,
                    dropout=0.02, act='idd', n_batch=30,
                    n_tbatch=30, K=35),
    'new_last-fm': dict(lr=0.0004, decay_rate=0.994, lamb=0.00014,
                        hidden_dim=48, attn_dim=5, n_layer=3,
                        dropout=0.02, act='idd', n_batch=36,
                        n_tbatch=36, K=50),
    'new_amazon-book': dict(lr=0.0005, decay_rate=0.994, lamb=0.000014,
                            hidden_dim=48, attn_dim=5, n_layer=3,
                            dropout=0.01, act='idd', n_batch=24,
                            n_tbatch=24, K=170),
    'amazon-book': dict(lr=0.0012, decay_rate=0.994, lamb=0.000014,
                        hidden_dim=48, attn_dim=5, n_layer=3,
                        dropout=0.02, act='idd', n_batch=20,
                        n_tbatch=20, K=120),
    'Dis_5fold_item': dict(lr=0.0005, decay_rate=0.994, lamb=0.00001,
                           hidden_dim=48, attn_dim=5, n_layer=5,
                           dropout=0.01, act='idd', n_batch=20,
                           n_tbatch=20, K=35),
    'Dis_5fold_user': dict(lr=0.001, decay_rate=0.994, lamb=0.00001,
                           hidden_dim=48, attn_dim=5, n_layer=3,
                           dropout=0.01, act='idd', n_batch=24,
                           n_tbatch=24, K=550),
}
_DEFAULT_CFG = dict(lr=0.0002, decay_rate=0.9938, lamb=0.0001,
                    hidden_dim=48, attn_dim=5, n_layer=3,
                    dropout=0.02, act='idd', n_batch=20,
                    n_tbatch=20, K=50)


class Options:
    pass


def make_opts(dataset_name, loader, K_override=None):
    opts = Options()
    cfg = DATASET_CONFIGS.get(dataset_name, _DEFAULT_CFG)
    for k, v in cfg.items():
        setattr(opts, k, v)
    if K_override is not None:
        opts.K = K_override
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes
    return opts


def _forward_no_sampling(self, q_sub, q_rel, hidden, edges, nodes,
                         id_layer, n_layer, old_nodes_new_idx):
    """GNNLayer.forward with the PPR top-K block stripped out.

    For middle layers we keep every edge that get_neighbors returned, so the
    subgraph grows unrestricted (the "Without sampling" baseline). Last layer
    still filters to item tails so PI/SI are well-defined.
    """
    sampled_nodes_idx = torch.gt(nodes[:, 1], -1) & torch.lt(nodes[:, 1], self.n_node + 1)

    t_nodes = nodes

    if id_layer == n_layer - 1:
        sampled_nodes_idx = torch.gt(nodes[:, 1], self.n_user - 1) & \
                            torch.lt(nodes[:, 1], self.n_user + self.n_item)
        item_tail_index = torch.gt(edges[:, 3], self.n_user - 1) & \
                          torch.lt(edges[:, 3], self.n_user + self.n_item)
        edges = edges[item_tail_index]
        nodes, tail_index = torch.unique(edges[:, [0, 3]], dim=0,
                                         sorted=True, return_inverse=True)
        edges = torch.cat([edges[:, 0:5], tail_index.unsqueeze(1)], 1)
        final_nodes = nodes
    else:
        final_nodes = torch.tensor([0])

    sub = edges[:, 4]
    rel = edges[:, 2]
    obj = edges[:, 5]

    hs = hidden[sub]
    hr = self.rela_embed(rel)
    r_idx = edges[:, 0]
    h_qr = self.rela_embed(q_rel)[r_idx]

    message = hs + hr
    alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(
        self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
    message = alpha * message

    message_agg = scatter(message, index=obj, dim=0,
                          dim_size=nodes.size(0), reduce='sum')
    hidden_new = self.act(self.W_h(message_agg))

    return hidden_new, t_nodes, final_nodes, old_nodes_new_idx, sampled_nodes_idx, alpha, edges


def _patch_no_sampling(model):
    saved = [layer.forward for layer in model.gnn_layers]
    for layer in model.gnn_layers:
        layer.forward = MethodType(_forward_no_sampling, layer)
    return saved


def _restore(model, saved):
    for layer, fn in zip(model.gnn_layers, saved):
        layer.forward = fn


@torch.no_grad()
def measure_one_batch(model, loader, subs, rels):
    """Run one inference pass and record per-layer node counts + final items.

    Returns:
        layer_node_counts: np.ndarray [n_layer, n_users_in_batch]
        final_items:       list[n_users_in_batch] of np.ndarray of item global ids
    """
    n = len(subs)
    n_layer = model.n_layer

    q_sub = torch.LongTensor(subs).cuda()
    q_rel = torch.LongTensor(rels).cuda()
    h0 = torch.zeros((1, n, model.hidden_dim)).cuda()
    nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
    hidden = torch.zeros(n, model.hidden_dim).cuda()

    layer_node_counts = np.zeros((n_layer, n), dtype=np.int64)
    final_items = [np.array([], dtype=np.int64) for _ in range(n)]

    for i in range(n_layer):
        nodes_np = nodes.data.cpu().numpy()
        t_nodes, edges, old_nodes_new_idx = loader.get_neighbors(nodes_np, mode='test')
        layer = model.gnn_layers[i]

        hidden, nodes, final_nodes, old_nodes_new_idx, sampled_nodes_idx, _, _ = layer(
            q_sub, q_rel, hidden, edges, t_nodes, i, n_layer, old_nodes_new_idx
        )

        batch_idx_cpu = nodes[:, 0].cpu().numpy()
        layer_node_counts[i] = np.bincount(batch_idx_cpu, minlength=n)

        if i == n_layer - 1:
            fn = final_nodes.cpu().numpy()
            for u_local in range(n):
                final_items[u_local] = fn[fn[:, 0] == u_local, 1]
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda() \
                .index_copy_(1, old_nodes_new_idx, h0)
            h0 = h0[0, sampled_nodes_idx, :].unsqueeze(0)
        else:
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda() \
                .index_copy_(1, old_nodes_new_idx, h0)

        hidden = model.dropout(hidden)
        hidden, h0 = model.gate(hidden.unsqueeze(0), h0)
        hidden = hidden.squeeze(0)

    return layer_node_counts, final_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--mode', choices=['model', 'none'], default='model',
                    help='"model"=use the checkpoint\'s sampling as trained; '
                         '"none"=bypass PPR top-K in middle layers.')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_users', type=int, default=None,
                    help='Cap on number of test users (recommended for --mode none).')
    ap.add_argument('--batch_size', type=int, default=None,
                    help='Override n_tbatch from config. --mode none defaults to 4.')
    ap.add_argument('--tag', type=str, default='',
                    help='Suffix appended to output filenames.')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    parts = [p for p in args.data_path.split('/') if p]
    dataset = parts[-1]

    print(f'[load] dataset={dataset}  ckpt={args.ckpt}  mode={args.mode}')
    loader = DataLoader(args.data_path)
    opts = make_opts(dataset, loader)

    model = KUCNet_trans(opts, loader).cuda()
    state = torch.load(args.ckpt, map_location='cuda')
    sd = state['model_state_dict'] if isinstance(state, dict) and 'model_state_dict' in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'[warn] missing state_dict keys: {missing}')
    if unexpected:
        print(f'[warn] unexpected state_dict keys: {unexpected}')
    model.eval()

    saved_forwards = _patch_no_sampling(model) if args.mode == 'none' else None

    n_test = loader.n_test
    if args.max_users is not None:
        n_test = min(n_test, args.max_users)

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = 4 if args.mode == 'none' else opts.n_tbatch
    n_batch = (n_test + batch_size - 1) // batch_size

    print(f'[run]  n_test={n_test}  batch_size={batch_size}  n_batch={n_batch}')

    all_layer_counts = []
    all_PI = []
    all_SI = []
    t0 = time.time()
    try:
        for b in range(n_batch):
            start = b * batch_size
            end = min(n_test, (b + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, _ = loader.get_batch(batch_idx, data='test')

            layer_counts, final_items = measure_one_batch(model, loader, subs, rels)

            for u_local, u_global in enumerate(subs):
                pos_set = set(int(x) for x in loader.test_user_set.get(int(u_global), []))
                items = final_items[u_local]
                SI = int(items.shape[0])
                PI = int(sum(1 for it in items if int(it) in pos_set))
                all_SI.append(SI)
                all_PI.append(PI)
            all_layer_counts.append(layer_counts)

            if b % 20 == 0 or b == n_batch - 1:
                print(f'  batch {b+1}/{n_batch}  users {start}-{end}')
    finally:
        if saved_forwards is not None:
            _restore(model, saved_forwards)

    elapsed = time.time() - t0
    all_layer_counts = np.concatenate(all_layer_counts, axis=1)
    n_users_eval = int(all_layer_counts.shape[1])
    avg_nodes = all_layer_counts.mean(axis=1)
    PI_avg = float(np.mean(all_PI))
    SI_avg = float(np.mean(all_SI))
    PoS = PI_avg / SI_avg if SI_avg > 0 else 0.0

    lines = [
        f'dataset={dataset}  mode={args.mode}  ckpt={args.ckpt}',
        f'n_users_eval={n_users_eval}  elapsed={elapsed:.2f}s  '
        f'avg_inference_per_user={elapsed/max(n_users_eval,1)*1000:.3f}ms',
    ]
    for i, c in enumerate(avg_nodes):
        lines.append(f'avg_nodes_layer_{i+1} = {c:.2f}')
    lines.append(f'PI(L) = {PI_avg:.4f}')
    lines.append(f'SI(L) = {SI_avg:.4f}')
    lines.append(f'PoS(L) = {PoS:.6e}')

    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    tag = ('_' + args.tag) if args.tag else ''
    out_txt = os.path.join(out_dir, f'{dataset}_subgraph_{args.mode}{tag}.txt')
    out_json = os.path.join(out_dir, f'{dataset}_subgraph_{args.mode}{tag}.json')

    print('\n'.join(lines))
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    summary = {
        'dataset': dataset,
        'mode': args.mode,
        'ckpt': args.ckpt,
        'n_users_eval': n_users_eval,
        'elapsed_sec': elapsed,
        'avg_nodes_per_layer': [float(x) for x in avg_nodes],
        'PI': PI_avg,
        'SI': SI_avg,
        'PoS': PoS,
    }
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nwrote {out_txt}\nwrote {out_json}')


if __name__ == '__main__':
    main()
