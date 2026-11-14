public class FastEngine 
{
    // Reusable buffers allocated ONCE to eliminate all GC overhead
    public static class Workspace
    {
        public double[][] x;
        public double[][][] all_q;
        public double[][][] all_k;
        public double[][][] all_v;
        public double[][] all_a;
        public double[] head_outs;
        public double[] net_in;
        public double[] h;
        public double[] probs;
        public double loss;

        // Backward scratch buffers
        public double[] g;
        public double[] dh;
        public double[] dz;
        public double[] d_net_in;
        public double[][] dx;
        public double[] d_out_h;
        public double[][] dv_h;
        public double[] da_h;
        public double[] ds_h;
        public double[][] dk_h;
        public double[] dq_h;

        // Scratch for softmax
        public double[] scores_h;
        public double[] exp_s;
        public double[] out_scores;
        public double[] exp_out;

        public Workspace(int n, int d, int n_heads, int h_size, int v_size)
        {
            int head_dim = d / n_heads;
            x = new double[n][d];
            all_q = new double[n_heads][1][head_dim];
            all_k = new double[n_heads][n][head_dim];
            all_v = new double[n_heads][n][head_dim];
            all_a = new double[n_heads][n];
            head_outs = new double[d];
            net_in = new double[d];
            h = new double[h_size];
            probs = new double[v_size];

            g = new double[v_size];
            dh = new double[h_size];
            dz = new double[h_size];
            d_net_in = new double[d];
            dx = new double[n][d];
            d_out_h = new double[head_dim];
            dv_h = new double[n][head_dim];
            da_h = new double[n];
            ds_h = new double[n];
            dk_h = new double[n][head_dim];
            dq_h = new double[head_dim];

            scores_h = new double[n];
            exp_s = new double[n];
            out_scores = new double[v_size];
            exp_out = new double[v_size];
        }
    }

    public static void forward_fast(int[] context, int target_idx,
                                    Library.ModelWeights mw, Workspace ws,
                                    int n, int d, int n_heads, int h_size, int v_size)
    {
        int head_dim = d / n_heads;
        double scale = Math.sqrt(head_dim);

        for(int i = 0; i < n; i++)
        {
            int char_row = context[i];
            for(int m = 0; m < d; m++)
                ws.x[i][m] = mw.C[char_row][m] + mw.P[i][m];
        }

        for(int m = 0; m < d; m++)
            ws.head_outs[m] = 0.0;

        for(int head = 0; head < n_heads; head++)
        {
            double[] q_h = ws.all_q[head][0];
            for(int col = 0; col < head_dim; col++)
            {
                double sum = 0.0;
                for(int row = 0; row < d; row++)
                    sum += ws.x[n - 1][row] * mw.Wq[head][row][col];
                q_h[col] = sum;
            }

            for(int i = 0; i < n; i++)
            {
                for(int col = 0; col < head_dim; col++)
                {
                    double sum_k = 0.0;
                    double sum_v = 0.0;
                    for(int row = 0; row < d; row++)
                    {
                        sum_k += ws.x[i][row] * mw.Wk[head][row][col];
                        sum_v += ws.x[i][row] * mw.Wv[head][row][col];
                    }
                    ws.all_k[head][i][col] = sum_k;
                    ws.all_v[head][i][col] = sum_v;
                }
            }

            double max_s = -Double.MAX_VALUE;
            for(int i = 0; i < n; i++)
            {
                double dot = 0.0;
                for(int m = 0; m < head_dim; m++)
                    dot += q_h[m] * ws.all_k[head][i][m];
                ws.scores_h[i] = dot / scale;
                if(ws.scores_h[i] > max_s)
                    max_s = ws.scores_h[i];
            }

            double sum_exp = 0.0;
            for(int i = 0; i < n; i++)
            {
                ws.exp_s[i] = Math.exp(ws.scores_h[i] - max_s);
                sum_exp += ws.exp_s[i];
            }

            for(int i = 0; i < n; i++)
                ws.all_a[head][i] = ws.exp_s[i] / sum_exp;

            int offset = head * head_dim;
            for(int m = 0; m < head_dim; m++)
            {
                double sum = 0.0;
                for(int i = 0; i < n; i++)
                    sum += ws.all_a[head][i] * ws.all_v[head][i][m];
                ws.head_outs[offset + m] = sum;
            }
        }

        // Additive residual: head_outs + x[n-1]
        for(int m = 0; m < d; m++)
            ws.net_in[m] = ws.head_outs[m] + ws.x[n - 1][m];

        // Hidden layer
        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < d; row++)
                sum += ws.net_in[row] * mw.W1[row][col];
            ws.h[col] = Math.tanh(sum + mw.b1[col]);
        }

        // Output layer
        double max_out = -Double.MAX_VALUE;
        for(int col = 0; col < v_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < h_size; row++)
                sum += ws.h[row] * mw.W2[row][col];
            ws.out_scores[col] = sum + mw.b2[col];
            if(ws.out_scores[col] > max_out)
                max_out = ws.out_scores[col];
        }

        double sum_exp_out = 0.0;
        for(int j = 0; j < v_size; j++)
        {
            ws.exp_out[j] = Math.exp(ws.out_scores[j] - max_out);
            sum_exp_out += ws.exp_out[j];
        }

        for(int j = 0; j < v_size; j++)
            ws.probs[j] = ws.exp_out[j] / sum_exp_out;

        if(target_idx >= 0 && target_idx < v_size)
            ws.loss = -Math.log(ws.probs[target_idx]);
        else
            ws.loss = 0.0;
    }

    public static void backward_fast(int[] context, int target_idx,
                                     Library.ModelWeights mw, Workspace ws,
                                     double lr, double lr_qk,
                                     int n, int d, int n_heads, int h_size, int v_size)
    {
        int head_dim = d / n_heads;
        double scale = Math.sqrt(head_dim);

        for(int j = 0; j < v_size; j++)
            ws.g[j] = ws.probs[j] - ((j == target_idx) ? 1.0 : 0.0);

        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int j = 0; j < v_size; j++)
                sum += mw.W2[col][j] * ws.g[j];
            ws.dh[col] = sum;
            ws.dz[col] = sum * (1.0 - (ws.h[col] * ws.h[col]));
        }

        for(int row = 0; row < d; row++)
        {
            double sum = 0.0;
            for(int col = 0; col < h_size; col++)
                sum += mw.W1[row][col] * ws.dz[col];
            ws.d_net_in[row] = sum;
        }

        // Update W2 and b2
        for(int row = 0; row < h_size; row++)
        {
            double h_val = ws.h[row];
            for(int col = 0; col < v_size; col++)
                mw.W2[row][col] -= lr * h_val * ws.g[col];
        }

        for(int col = 0; col < v_size; col++)
            mw.b2[col] -= lr * ws.g[col];

        // Update W1 and b1
        for(int row = 0; row < d; row++)
        {
            double in_val = ws.net_in[row];
            for(int col = 0; col < h_size; col++)
                mw.W1[row][col] -= lr * in_val * ws.dz[col];
        }

        for(int col = 0; col < h_size; col++)
            mw.b1[col] -= lr * ws.dz[col];

        // Initialize dx
        for(int i = 0; i < n; i++)
            for(int m = 0; m < d; m++)
                ws.dx[i][m] = (i == n - 1) ? ws.d_net_in[m] : 0.0;

        for(int head = 0; head < n_heads; head++)
        {
            int offset = head * head_dim;
            for(int m = 0; m < head_dim; m++)
                ws.d_out_h[m] = ws.d_net_in[offset + m];

            double sum_a_da = 0.0;
            for(int i = 0; i < n; i++)
            {
                double dot = 0.0;
                for(int m = 0; m < head_dim; m++)
                {
                    ws.dv_h[i][m] = ws.all_a[head][i] * ws.d_out_h[m];
                    dot += ws.d_out_h[m] * ws.all_v[head][i][m];
                }
                ws.da_h[i] = dot;
                sum_a_da += ws.all_a[head][i] * dot;
            }

            for(int m = 0; m < head_dim; m++)
                ws.dq_h[m] = 0.0;

            for(int i = 0; i < n; i++)
            {
                ws.ds_h[i] = ws.all_a[head][i] * (ws.da_h[i] - sum_a_da);
                double s_scaled = ws.ds_h[i] / scale;
                for(int m = 0; m < head_dim; m++)
                {
                    ws.dk_h[i][m] = s_scaled * ws.all_q[head][0][m];
                    ws.dq_h[m] += s_scaled * ws.all_k[head][i][m];
                }
            }

            // Accumulate dx
            for(int i = 0; i < n; i++)
            {
                for(int row = 0; row < d; row++)
                {
                    double sum_v = 0.0;
                    double sum_k = 0.0;
                    for(int col = 0; col < head_dim; col++)
                    {
                        sum_v += ws.dv_h[i][col] * mw.Wv[head][row][col];
                        sum_k += ws.dk_h[i][col] * mw.Wk[head][row][col];
                    }
                    ws.dx[i][row] += sum_v + sum_k;
                }
            }

            for(int row = 0; row < d; row++)
            {
                double sum_q = 0.0;
                for(int col = 0; col < head_dim; col++)
                    sum_q += ws.dq_h[col] * mw.Wq[head][row][col];
                ws.dx[n - 1][row] += sum_q;
            }

            // Update head weights
            double[] x_last = ws.x[n - 1];
            for(int row = 0; row < d; row++)
            {
                double x_r = x_last[row];
                for(int col = 0; col < head_dim; col++)
                {
                    mw.Wq[head][row][col] -= lr_qk * x_r * ws.dq_h[col];
                    double sum_wk = 0.0;
                    double sum_wv = 0.0;
                    for(int i = 0; i < n; i++)
                    {
                        sum_wk += ws.x[i][row] * ws.dk_h[i][col];
                        sum_wv += ws.x[i][row] * ws.dv_h[i][col];
                    }
                    mw.Wk[head][row][col] -= lr_qk * sum_wk;
                    mw.Wv[head][row][col] -= lr * sum_wv;
                }
            }
        }

        // Update C and P
        for(int i = 0; i < n; i++)
        {
            int char_row = context[i];
            for(int m = 0; m < d; m++)
            {
                double dx_val = ws.dx[i][m];
                mw.C[char_row][m] -= lr * dx_val;
                mw.P[i][m] -= lr * dx_val;
            }
        }
    }
}
