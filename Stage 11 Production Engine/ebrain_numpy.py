import os
import math
import struct
import numpy as np

class Vocab:
    UNK_CHAR = '\u25A1'

    def __init__(self, characters):
        self.itos = list(characters)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def build_from_text(cls, text, min_freq=2):
        from collections import Counter
        counts = Counter(text)
        chars = [cls.UNK_CHAR]
        for ch, count in counts.most_common():
            if count >= min_freq and ch != cls.UNK_CHAR:
                chars.append(ch)
        return cls(chars)

    def encode(self, ch):
        return self.stoi.get(ch, 0)

    def decode(self, idx):
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return self.UNK_CHAR

    def encode_string(self, s):
        return np.array([self.encode(ch) for ch in s], dtype=np.int32)

    def decode_tokens(self, tokens):
        return "".join(self.decode(int(t)) for t in tokens)

    def size(self):
        return len(self.itos)


class EBrainNumpy:
    """
    Pure NumPy handcrafted reference implementation of E-Brain Causal Transformer.
    Direct mathematical 1:1 mirror of the Java TransformerEngine.
    """
    def __init__(self, V, T=16, D=32, n_heads=4, H_mlp=64, L=2, use_layernorm=True, init_scale=0.02, seed=42):
        self.V = V
        self.T = T
        self.D = D
        self.n_heads = n_heads
        self.dk = D // n_heads
        self.H_mlp = H_mlp
        self.L = L
        self.use_layernorm = use_layernorm

        rng = np.random.RandomState(seed)
        self.C = rng.uniform(-init_scale, init_scale, (V, D)).astype(np.float64)
        self.P = rng.uniform(-init_scale, init_scale, (T, D)).astype(np.float64)

        self.blocks = []
        for _ in range(L):
            b = {
                'Wq': rng.uniform(-init_scale, init_scale, (n_heads, D, self.dk)).astype(np.float64),
                'Wk': rng.uniform(-init_scale, init_scale, (n_heads, D, self.dk)).astype(np.float64),
                'Wv': rng.uniform(-init_scale, init_scale, (n_heads, D, self.dk)).astype(np.float64),
                'Wo': rng.uniform(-init_scale, init_scale, (D, D)).astype(np.float64),
                'W1': rng.uniform(-init_scale, init_scale, (D, H_mlp)).astype(np.float64),
                'b1': np.zeros(H_mlp, dtype=np.float64),
                'W2': rng.uniform(-init_scale, init_scale, (H_mlp, D)).astype(np.float64),
                'b2': np.zeros(D, dtype=np.float64),
                'gamma1': np.ones(D, dtype=np.float64),
                'beta1': np.zeros(D, dtype=np.float64),
                'gamma2': np.ones(D, dtype=np.float64),
                'beta2': np.zeros(D, dtype=np.float64),
            }
            self.blocks.append(b)

        self.gammaF = np.ones(D, dtype=np.float64)
        self.betaF = np.zeros(D, dtype=np.float64)
        self.Wout = rng.uniform(-init_scale, init_scale, (D, V)).astype(np.float64)
        self.bout = np.zeros(V, dtype=np.float64)

        # Adam moment buffers
        self.step_count = 0
        self.mC = np.zeros_like(self.C); self.vC = np.zeros_like(self.C)
        self.mP = np.zeros_like(self.P); self.vP = np.zeros_like(self.P)
        self.mWout = np.zeros_like(self.Wout); self.vWout = np.zeros_like(self.Wout)
        self.mbout = np.zeros_like(self.bout); self.vbout = np.zeros_like(self.bout)
        self.mGammaF = np.zeros_like(self.gammaF); self.vGammaF = np.zeros_like(self.gammaF)
        self.mBetaF = np.zeros_like(self.betaF); self.vBetaF = np.zeros_like(self.betaF)

        self.block_moments = []
        for _ in range(L):
            bm = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in self.blocks[0].items()}
            self.block_moments.append(bm)

    @staticmethod
    def _layer_norm_forward(x, gamma, beta, eps=1e-5):
        # x: [T, D]
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + eps)
        x_hat = (x - mean) * std_inv
        y = gamma * x_hat + beta if gamma is not None else x_hat
        return y, x_hat, std_inv

    @staticmethod
    def _layer_norm_backward(dy, x_hat, std_inv, gamma):
        # dy: [T, D]
        D = dy.shape[-1]
        dgamma = np.sum(dy * x_hat, axis=0) if gamma is not None else None
        dbeta = np.sum(dy, axis=0) if gamma is not None else None
        dx_hat = dy * gamma if gamma is not None else dy
        sum_dx_hat = np.sum(dx_hat, axis=-1, keepdims=True)
        sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        dx = (std_inv / D) * (D * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat)
        return dx, dgamma, dbeta

    def forward(self, tokens, targets=None, is_training=False, dropout_rate=0.0):
        # tokens: [T]
        T = len(tokens)
        D = self.D
        cache = {'tokens': tokens, 'targets': targets, 'blocks': []}

        # 1. Embeddings
        emb = self.C[tokens] + self.P[:T]
        cache['emb'] = emb
        cur_X = emb

        sqrt_dk = math.sqrt(self.dk)

        for l in range(self.L):
            b = self.blocks[l]
            b_cache = {'X': cur_X}

            # LN 1
            if self.use_layernorm:
                norm_X1, x_hat1, std_inv1 = self._layer_norm_forward(cur_X, b['gamma1'], b['beta1'])
                b_cache['ln1'] = (norm_X1, x_hat1, std_inv1)
            else:
                norm_X1 = cur_X

            # Multi-head Attention
            # Q, K, V: [H, T, dk]
            Q = np.zeros((self.n_heads, T, self.dk), dtype=np.float64)
            K = np.zeros((self.n_heads, T, self.dk), dtype=np.float64)
            V = np.zeros((self.n_heads, T, self.dk), dtype=np.float64)
            for h in range(self.n_heads):
                Q[h] = norm_X1 @ b['Wq'][h]
                K[h] = norm_X1 @ b['Wk'][h]
                V[h] = norm_X1 @ b['Wv'][h]

            b_cache['Q'] = Q; b_cache['K'] = K; b_cache['V'] = V

            # Attention scores with causal mask
            scores = (Q @ K.swapaxes(1, 2)) / sqrt_dk # [H, T, T]
            # Causal mask: upper triangle set to -1e9
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores[:, mask] = -1e9
            b_cache['scores'] = scores

            # Softmax
            max_s = np.max(scores, axis=-1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            exp_s[:, mask] = 0.0
            A = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
            b_cache['A'] = A

            # Attn output: [H, T, dk] -> concat to [T, D]
            head_out = A @ V # [H, T, dk]
            head_concat = head_out.swapaxes(0, 1).reshape(T, D) # [T, D]
            b_cache['head_concat'] = head_concat

            att_out = head_concat @ b['Wo']
            b_cache['att_out'] = att_out
            Y = cur_X + att_out # Additive residual
            b_cache['Y'] = Y

            # LN 2
            if self.use_layernorm:
                norm_Y, y_hat2, std_inv2 = self._layer_norm_forward(Y, b['gamma2'], b['beta2'])
                b_cache['ln2'] = (norm_Y, y_hat2, std_inv2)
            else:
                norm_Y = Y

            # Feed-Forward: W1 -> tanh -> W2
            h_raw = norm_Y @ b['W1'] + b['b1']
            h_tanh = np.tanh(h_raw)
            if is_training and dropout_rate > 0.0:
                mask_drop = (np.random.rand(*h_tanh.shape) >= dropout_rate).astype(np.float64) / (1.0 - dropout_rate)
                h_drop = h_tanh * mask_drop
            else:
                mask_drop = np.ones_like(h_tanh)
                h_drop = h_tanh

            b_cache['h_raw'] = h_raw
            b_cache['h_tanh'] = h_tanh
            b_cache['mask_drop'] = mask_drop
            b_cache['h_drop'] = h_drop

            ff_out = h_drop @ b['W2'] + b['b2']
            b_cache['ff_out'] = ff_out
            Z = Y + ff_out # Additive residual
            b_cache['Z'] = Z

            cache['blocks'].append(b_cache)
            cur_X = Z

        # Final LN
        if self.use_layernorm:
            final_features, f_hat, std_inv_f = self._layer_norm_forward(cur_X, self.gammaF, self.betaF)
            cache['lnF'] = (final_features, f_hat, std_inv_f)
        else:
            final_features = cur_X

        # Output head
        logits = final_features @ self.Wout + self.bout
        cache['logits'] = logits

        max_l = np.max(logits, axis=-1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        cache['probs'] = probs

        if targets is not None:
            # Cross-entropy loss
            target_probs = probs[np.arange(T), targets]
            losses = -np.log(np.maximum(target_probs, 1e-15))
            cache['losses'] = losses
            cache['avg_loss'] = float(np.mean(losses))

        return cache

    def generate(self, vocab, prompt, gen_len=200, temperature=0.8):
        prompt_tokens = list(vocab.encode_string(prompt))
        T = self.T
        if len(prompt_tokens) < T:
            window = [vocab.encode(' ')] * (T - len(prompt_tokens)) + prompt_tokens
        else:
            window = prompt_tokens[-T:]

        out_tokens = list(prompt_tokens)
        for _ in range(gen_len):
            cache = self.forward(np.array(window, dtype=np.int32), is_training=False)
            last_logits = cache['logits'][-1] / max(temperature, 1e-4)
            max_l = np.max(last_logits)
            p = np.exp(last_logits - max_l)
            p /= np.sum(p)

            next_tok = int(np.random.choice(self.V, p=p))
            out_tokens.append(next_tok)
            window = window[1:] + [next_tok]

        return vocab.decode_tokens(out_tokens)
