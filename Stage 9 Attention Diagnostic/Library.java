import java.util.concurrent.ThreadLocalRandom;

public class Library 
{
    static char[] ch_present;

    public static void fill_the_characters_present_in_text(String data)
    {
        char[] temp = new char[256];
        int c = 0;
        for(int i = 0; i < data.length(); i++)
        {
            char ch = data.charAt(i);
            boolean found = false;
            for(int j = 0; j < c; j++)
            {
                if(temp[j] == ch)
                {
                    found = true;
                    break;
                }
            }
            if(!found)
            {
                temp[c] = ch;
                c++;
            }
        }

        ch_present = new char[c];
        for(int i = 0; i < c; i++)
        {
            ch_present[i] = temp[i];
        }
    }

    public static int find_ch_number(char ch, char[] array)
    {
        for(int i = 0; i < array.length; i++)
        {
            if(array[i] == ch)
                return i;
        }
        return -1;
    }

    public static Character find_ch_at_a_number(int n, char[] array)
    {
        if(n < 0 || n >= array.length)
            return null;
        return array[n];
    }

    public static double[][] make_random_matrix(int rows, int cols, double scale)
    {
        double[][] mat = new double[rows][cols];
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                mat[i][j] = ThreadLocalRandom.current().nextDouble(-scale, scale);
        return mat;
    }

    public static double[] make_zeros_vector(int len)
    {
        return new double[len];
    }

    public static double[][] copy_2d(double[][] a)
    {
        double[][] b = new double[a.length][];
        for(int i = 0; i < a.length; i++)
            b[i] = a[i].clone();
        return b;
    }

    public static double[][][] copy_3d(double[][][] a)
    {
        double[][][] b = new double[a.length][][];
        for(int i = 0; i < a.length; i++)
            b[i] = copy_2d(a[i]);
        return b;
    }

    public static double[] copy_1d(double[] a)
    {
        return a.clone();
    }

    // =========================================================================
    // MULTI-HEAD ATTENTION MODEL STATE
    // =========================================================================
    public static class ModelWeights
    {
        public double[][] C;
        public double[][] P;
        public double[][][] Wq;
        public double[][][] Wk;
        public double[][][] Wv;
        public double[][] W1;
        public double[] b1;
        public double[][] W2;
        public double[] b2;

        public ModelWeights copy()
        {
            ModelWeights mw = new ModelWeights();
            mw.C = copy_2d(this.C);
            mw.P = copy_2d(this.P);
            mw.Wq = copy_3d(this.Wq);
            mw.Wk = copy_3d(this.Wk);
            mw.Wv = copy_3d(this.Wv);
            mw.W1 = copy_2d(this.W1);
            mw.b1 = copy_1d(this.b1);
            mw.W2 = copy_2d(this.W2);
            mw.b2 = copy_1d(this.b2);
            return mw;
        }

        public void restore_from(ModelWeights src)
        {
            this.C = copy_2d(src.C);
            this.P = copy_2d(src.P);
            this.Wq = copy_3d(src.Wq);
            this.Wk = copy_3d(src.Wk);
            this.Wv = copy_3d(src.Wv);
            this.W1 = copy_2d(src.W1);
            this.b1 = copy_1d(src.b1);
            this.W2 = copy_2d(src.W2);
            this.b2 = copy_1d(src.b2);
        }
    }

    public static ModelWeights init_model(int V, int N, int D, int n_heads, int H, double init_scale, double qk_init_scale)
    {
        ModelWeights mw = new ModelWeights();
        int head_dim = D / n_heads;

        mw.C = make_random_matrix(V, D, init_scale);
        mw.P = make_random_matrix(N, D, init_scale);

        mw.Wq = new double[n_heads][D][head_dim];
        mw.Wk = new double[n_heads][D][head_dim];
        mw.Wv = new double[n_heads][D][head_dim];
        for(int h = 0; h < n_heads; h++)
        {
            mw.Wq[h] = make_random_matrix(D, head_dim, qk_init_scale);
            mw.Wk[h] = make_random_matrix(D, head_dim, qk_init_scale);
            mw.Wv[h] = make_random_matrix(D, head_dim, init_scale);
        }

        mw.W1 = make_random_matrix(D, H, init_scale);
        mw.b1 = make_zeros_vector(H);
        mw.W2 = make_random_matrix(H, V, init_scale);
        mw.b2 = make_zeros_vector(V);
        return mw;
    }

    public static class MultiHeadCache
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
    }

    public static MultiHeadCache forward_multi_head(int[] context, int target_idx,
                                                    ModelWeights mw,
                                                    int n, int d, int n_heads, int h_size, int v_size)
    {
        MultiHeadCache mc = new MultiHeadCache();
        int head_dim = d / n_heads;
        double scale = Math.sqrt(head_dim);

        mc.x = new double[n][d];
        for(int i = 0; i < n; i++)
        {
            int char_row = context[i];
            for(int m = 0; m < d; m++)
                mc.x[i][m] = mw.C[char_row][m] + mw.P[i][m];
        }

        mc.all_q = new double[n_heads][1][head_dim];
        mc.all_k = new double[n_heads][n][head_dim];
        mc.all_v = new double[n_heads][n][head_dim];
        mc.all_a = new double[n_heads][n];
        mc.head_outs = new double[d];

        for(int h = 0; h < n_heads; h++)
        {
            double[] q_h = new double[head_dim];
            for(int col = 0; col < head_dim; col++)
            {
                double sum = 0.0;
                for(int row = 0; row < d; row++)
                    sum = sum + (mc.x[n - 1][row] * mw.Wq[h][row][col]);
                q_h[col] = sum;
            }
            mc.all_q[h][0] = q_h;

            for(int i = 0; i < n; i++)
            {
                for(int col = 0; col < head_dim; col++)
                {
                    double sum_k = 0.0;
                    double sum_v = 0.0;
                    for(int row = 0; row < d; row++)
                    {
                        sum_k = sum_k + (mc.x[i][row] * mw.Wk[h][row][col]);
                        sum_v = sum_v + (mc.x[i][row] * mw.Wv[h][row][col]);
                    }
                    mc.all_k[h][i][col] = sum_k;
                    mc.all_v[h][i][col] = sum_v;
                }
            }

            double[] scores_h = new double[n];
            double max_s = -Double.MAX_VALUE;
            for(int i = 0; i < n; i++)
            {
                double dot = 0.0;
                for(int m = 0; m < head_dim; m++)
                    dot = dot + (q_h[m] * mc.all_k[h][i][m]);
                scores_h[i] = dot / scale;
                if(scores_h[i] > max_s)
                    max_s = scores_h[i];
            }

            double sum_exp = 0.0;
            double[] exp_s = new double[n];
            for(int i = 0; i < n; i++)
            {
                exp_s[i] = Math.exp(scores_h[i] - max_s);
                sum_exp = sum_exp + exp_s[i];
            }

            for(int i = 0; i < n; i++)
                mc.all_a[h][i] = exp_s[i] / sum_exp;

            for(int m = 0; m < head_dim; m++)
            {
                double sum = 0.0;
                for(int i = 0; i < n; i++)
                    sum = sum + (mc.all_a[h][i] * mc.all_v[h][i][m]);
                mc.head_outs[h * head_dim + m] = sum;
            }
        }

        // Additive residual: head_outs + x[n-1]
        mc.net_in = new double[d];
        for(int m = 0; m < d; m++)
            mc.net_in[m] = mc.head_outs[m] + mc.x[n - 1][m];

        // Hidden layer
        mc.h = new double[h_size];
        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < d; row++)
                sum = sum + (mc.net_in[row] * mw.W1[row][col]);
            mc.h[col] = Math.tanh(sum + mw.b1[col]);
        }

        // Output layer
        double[] out_scores = new double[v_size];
        double max_out = -Double.MAX_VALUE;
        for(int col = 0; col < v_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < h_size; row++)
                sum = sum + (mc.h[row] * mw.W2[row][col]);
            out_scores[col] = sum + mw.b2[col];
            if(out_scores[col] > max_out)
                max_out = out_scores[col];
        }

        double sum_exp_out = 0.0;
        double[] exp_out = new double[v_size];
        for(int j = 0; j < v_size; j++)
        {
            exp_out[j] = Math.exp(out_scores[j] - max_out);
            sum_exp_out = sum_exp_out + exp_out[j];
        }

        mc.probs = new double[v_size];
        for(int j = 0; j < v_size; j++)
            mc.probs[j] = exp_out[j] / sum_exp_out;

        if(target_idx >= 0 && target_idx < v_size)
            mc.loss = -Math.log(mc.probs[target_idx]);
        else
            mc.loss = 0.0;

        return mc;
    }

    public static void backward_multi_head(MultiHeadCache mc, int[] context, int target_idx,
                                           ModelWeights mw,
                                           double lr, double lr_qk,
                                           int n, int d, int n_heads, int h_size, int v_size)
    {
        int head_dim = d / n_heads;
        double scale = Math.sqrt(head_dim);

        double[] g = new double[v_size];
        for(int j = 0; j < v_size; j++)
            g[j] = mc.probs[j] - ((j == target_idx) ? 1.0 : 0.0);

        double[] dh = new double[h_size];
        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int j = 0; j < v_size; j++)
                sum = sum + (mw.W2[col][j] * g[j]);
            dh[col] = sum;
        }

        double[] dz = new double[h_size];
        for(int col = 0; col < h_size; col++)
            dz[col] = dh[col] * (1.0 - (mc.h[col] * mc.h[col]));

        double[] d_net_in = new double[d];
        for(int row = 0; row < d; row++)
        {
            double sum = 0.0;
            for(int col = 0; col < h_size; col++)
                sum = sum + (mw.W1[row][col] * dz[col]);
            d_net_in[row] = sum;
        }

        // Update W2 and b2
        for(int row = 0; row < h_size; row++)
            for(int col = 0; col < v_size; col++)
                mw.W2[row][col] = mw.W2[row][col] - (lr * mc.h[row] * g[col]);

        for(int col = 0; col < v_size; col++)
            mw.b2[col] = mw.b2[col] - (lr * g[col]);

        // Update W1 and b1
        for(int row = 0; row < d; row++)
            for(int col = 0; col < h_size; col++)
                mw.W1[row][col] = mw.W1[row][col] - (lr * mc.net_in[row] * dz[col]);

        for(int col = 0; col < h_size; col++)
            mw.b1[col] = mw.b1[col] - (lr * dz[col]);

        // Gradients into input x: initialize with residual addition
        double[][] dx = new double[n][d];
        for(int m = 0; m < d; m++)
            dx[n - 1][m] = d_net_in[m];

        // Process each head
        for(int h = 0; h < n_heads; h++)
        {
            double[] d_out_h = new double[head_dim];
            for(int m = 0; m < head_dim; m++)
                d_out_h[m] = d_net_in[h * head_dim + m];

            double[][] dv_h = new double[n][head_dim];
            for(int i = 0; i < n; i++)
                for(int m = 0; m < head_dim; m++)
                    dv_h[i][m] = mc.all_a[h][i] * d_out_h[m];

            double[] da_h = new double[n];
            double sum_a_da = 0.0;
            for(int i = 0; i < n; i++)
            {
                double dot = 0.0;
                for(int m = 0; m < head_dim; m++)
                    dot = dot + (d_out_h[m] * mc.all_v[h][i][m]);
                da_h[i] = dot;
                sum_a_da = sum_a_da + (mc.all_a[h][i] * da_h[i]);
            }

            double[] ds_h = new double[n];
            for(int i = 0; i < n; i++)
                ds_h[i] = mc.all_a[h][i] * (da_h[i] - sum_a_da);

            double[][] dk_h = new double[n][head_dim];
            double[] dq_h = new double[head_dim];
            for(int i = 0; i < n; i++)
            {
                double s_scaled = ds_h[i] / scale;
                for(int m = 0; m < head_dim; m++)
                {
                    dk_h[i][m] = s_scaled * mc.all_q[h][0][m];
                    dq_h[m] = dq_h[m] + (s_scaled * mc.all_k[h][i][m]);
                }
            }

            // Accumulate dx from head h
            for(int i = 0; i < n; i++)
            {
                for(int row = 0; row < d; row++)
                {
                    double sum_v = 0.0;
                    double sum_k = 0.0;
                    for(int col = 0; col < head_dim; col++)
                    {
                        sum_v = sum_v + (dv_h[i][col] * mw.Wv[h][row][col]);
                        sum_k = sum_k + (dk_h[i][col] * mw.Wk[h][row][col]);
                    }
                    dx[i][row] = dx[i][row] + sum_v + sum_k;
                }
            }

            for(int row = 0; row < d; row++)
            {
                double sum_q = 0.0;
                for(int col = 0; col < head_dim; col++)
                    sum_q = sum_q + (dq_h[col] * mw.Wq[h][row][col]);
                dx[n - 1][row] = dx[n - 1][row] + sum_q;
            }

            // Update head h weights Wq, Wk (using lr_qk), and Wv (using lr)
            for(int row = 0; row < d; row++)
            {
                for(int col = 0; col < head_dim; col++)
                {
                    mw.Wq[h][row][col] = mw.Wq[h][row][col] - (lr_qk * mc.x[n - 1][row] * dq_h[col]);
                    double sum_wk = 0.0;
                    double sum_wv = 0.0;
                    for(int i = 0; i < n; i++)
                    {
                        sum_wk = sum_wk + (mc.x[i][row] * dk_h[i][col]);
                        sum_wv = sum_wv + (mc.x[i][row] * dv_h[i][col]);
                    }
                    mw.Wk[h][row][col] = mw.Wk[h][row][col] - (lr_qk * sum_wk);
                    mw.Wv[h][row][col] = mw.Wv[h][row][col] - (lr * sum_wv);
                }
            }
        }

        // Update C and P
        for(int i = 0; i < n; i++)
        {
            int char_row = context[i];
            for(int m = 0; m < d; m++)
            {
                mw.C[char_row][m] = mw.C[char_row][m] - (lr * dx[i][m]);
                mw.P[i][m] = mw.P[i][m] - (lr * dx[i][m]);
            }
        }
    }

    public static double compute_val_loss(String val_text, char[] vocab, ModelWeights mw,
                                          int n, int d, int n_heads, int h_size, int v_size)
    {
        int val_len = val_text.length() - n;
        double sum = 0.0;
        int[] ctx = new int[n];
        for(int t = 0; t < val_len; t++)
        {
            for(int i = 0; i < n; i++)
                ctx[i] = find_ch_number(val_text.charAt(t + i), vocab);
            int target = find_ch_number(val_text.charAt(t + n), vocab);
            MultiHeadCache mc = forward_multi_head(ctx, target, mw, n, d, n_heads, h_size, v_size);
            sum += mc.loss;
        }
        return sum / val_len;
    }
}
