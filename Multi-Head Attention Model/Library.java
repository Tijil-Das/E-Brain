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

    public static double relative_error(double a, double n)
    {
        return Math.abs(a - n) / (Math.abs(a) + Math.abs(n) + 1e-12);
    }

    // =========================================================================
    // 1. SINGLE-HEAD ATTENTION (Supports Joined or Additive Residual)
    // =========================================================================
    public static class SingleHeadCache
    {
        public double[][] x;
        public double[] q;
        public double[][] k;
        public double[][] v;
        public double[] a;
        public double[] att_out;
        public double[] net_in;
        public double[] h;
        public double[] probs;
        public double loss;
    }

    public static SingleHeadCache forward_single_head(int[] context, int target_idx,
                                                      double[][] C, double[][] P,
                                                      double[][] Wq, double[][] Wk, double[][] Wv,
                                                      double[][] W1, double[] b1, double[][] W2, double[] b2,
                                                      int n, int d, int h_size, int v_size, boolean additive_residual)
    {
        SingleHeadCache fc = new SingleHeadCache();
        double scale = Math.sqrt(d);

        fc.x = new double[n][d];
        for(int i = 0; i < n; i++)
        {
            int char_row = context[i];
            for(int m = 0; m < d; m++)
                fc.x[i][m] = C[char_row][m] + P[i][m];
        }

        fc.q = new double[d];
        for(int col = 0; col < d; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < d; row++)
                sum = sum + (fc.x[n - 1][row] * Wq[row][col]);
            fc.q[col] = sum;
        }

        fc.k = new double[n][d];
        fc.v = new double[n][d];
        for(int i = 0; i < n; i++)
        {
            for(int col = 0; col < d; col++)
            {
                double sum_k = 0.0;
                double sum_v = 0.0;
                for(int row = 0; row < d; row++)
                {
                    sum_k = sum_k + (fc.x[i][row] * Wk[row][col]);
                    sum_v = sum_v + (fc.x[i][row] * Wv[row][col]);
                }
                fc.k[i][col] = sum_k;
                fc.v[i][col] = sum_v;
            }
        }

        double[] scores = new double[n];
        double max_s = -Double.MAX_VALUE;
        for(int i = 0; i < n; i++)
        {
            double dot = 0.0;
            for(int m = 0; m < d; m++)
                dot = dot + (fc.q[m] * fc.k[i][m]);
            scores[i] = dot / scale;
            if(scores[i] > max_s)
                max_s = scores[i];
        }

        double sum_exp = 0.0;
        double[] exp_s = new double[n];
        for(int i = 0; i < n; i++)
        {
            exp_s[i] = Math.exp(scores[i] - max_s);
            sum_exp = sum_exp + exp_s[i];
        }

        fc.a = new double[n];
        for(int i = 0; i < n; i++)
            fc.a[i] = exp_s[i] / sum_exp;

        fc.att_out = new double[d];
        for(int m = 0; m < d; m++)
        {
            double sum = 0.0;
            for(int i = 0; i < n; i++)
                sum = sum + (fc.a[i] * fc.v[i][m]);
            fc.att_out[m] = sum;
        }

        //Residual Connection: Additive (D) or Joined (2D)
        if(additive_residual)
        {
            fc.net_in = new double[d];
            for(int m = 0; m < d; m++)
                fc.net_in[m] = fc.att_out[m] + fc.x[n - 1][m];
        }
        else
        {
            fc.net_in = new double[2 * d];
            for(int m = 0; m < d; m++)
            {
                fc.net_in[m] = fc.att_out[m];
                fc.net_in[d + m] = fc.x[n - 1][m];
            }
        }

        int in_dim = fc.net_in.length;
        fc.h = new double[h_size];
        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < in_dim; row++)
                sum = sum + (fc.net_in[row] * W1[row][col]);
            fc.h[col] = Math.tanh(sum + b1[col]);
        }

        double[] out_scores = new double[v_size];
        double max_out = -Double.MAX_VALUE;
        for(int col = 0; col < v_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < h_size; row++)
                sum = sum + (fc.h[row] * W2[row][col]);
            out_scores[col] = sum + b2[col];
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

        fc.probs = new double[v_size];
        for(int j = 0; j < v_size; j++)
            fc.probs[j] = exp_out[j] / sum_exp_out;

        fc.loss = -Math.log(fc.probs[target_idx]);
        return fc;
    }

    public static void backward_single_head(SingleHeadCache fc, int[] context, int target_idx,
                                            double[][] C, double[][] P,
                                            double[][] Wq, double[][] Wk, double[][] Wv,
                                            double[][] W1, double[] b1, double[][] W2, double[] b2,
                                            double lr, int n, int d, int h_size, int v_size, boolean additive_residual)
    {
        double scale = Math.sqrt(d);
        int in_dim = fc.net_in.length;

        double[] g = new double[v_size];
        for(int j = 0; j < v_size; j++)
            g[j] = fc.probs[j] - ((j == target_idx) ? 1.0 : 0.0);

        double[] dh = new double[h_size];
        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int j = 0; j < v_size; j++)
                sum = sum + (W2[col][j] * g[j]);
            dh[col] = sum;
        }

        double[] dz = new double[h_size];
        for(int col = 0; col < h_size; col++)
            dz[col] = dh[col] * (1.0 - (fc.h[col] * fc.h[col]));

        double[] d_net_in = new double[in_dim];
        for(int row = 0; row < in_dim; row++)
        {
            double sum = 0.0;
            for(int col = 0; col < h_size; col++)
                sum = sum + (W1[row][col] * dz[col]);
            d_net_in[row] = sum;
        }

        double[] d_att_out = new double[d];
        double[] dx_res = new double[d];
        if(additive_residual)
        {
            for(int m = 0; m < d; m++)
            {
                d_att_out[m] = d_net_in[m];
                dx_res[m] = d_net_in[m]; // flowing equally into both branches of addition
            }
        }
        else
        {
            for(int m = 0; m < d; m++)
            {
                d_att_out[m] = d_net_in[m];
                dx_res[m] = d_net_in[d + m];
            }
        }

        double[][] dv = new double[n][d];
        for(int i = 0; i < n; i++)
            for(int m = 0; m < d; m++)
                dv[i][m] = fc.a[i] * d_att_out[m];

        double[] da = new double[n];
        double sum_a_da = 0.0;
        for(int i = 0; i < n; i++)
        {
            double dot = 0.0;
            for(int m = 0; m < d; m++)
                dot = dot + (d_att_out[m] * fc.v[i][m]);
            da[i] = dot;
            sum_a_da = sum_a_da + (fc.a[i] * da[i]);
        }

        double[] ds = new double[n];
        for(int i = 0; i < n; i++)
            ds[i] = fc.a[i] * (da[i] - sum_a_da);

        double[][] dk = new double[n][d];
        double[] dq = new double[d];
        for(int i = 0; i < n; i++)
        {
            double s_scaled = ds[i] / scale;
            for(int m = 0; m < d; m++)
            {
                dk[i][m] = s_scaled * fc.q[m];
                dq[m] = dq[m] + (s_scaled * fc.k[i][m]);
            }
        }

        //Update W2 and b2
        for(int j = 0; j < v_size; j++)
        {
            b2[j] = b2[j] - (lr * g[j]);
            for(int col = 0; col < h_size; col++)
                W2[col][j] = W2[col][j] - (lr * fc.h[col] * g[j]);
        }

        //Update W1 and b1
        for(int col = 0; col < h_size; col++)
        {
            b1[col] = b1[col] - (lr * dz[col]);
            for(int row = 0; row < in_dim; row++)
                W1[row][col] = W1[row][col] - (lr * fc.net_in[row] * dz[col]);
        }

        double[][] dx = new double[n][d];
        for(int i = 0; i < n; i++)
        {
            for(int row = 0; row < d; row++)
            {
                double sum_v = 0.0;
                double sum_k = 0.0;
                for(int col = 0; col < d; col++)
                {
                    sum_v = sum_v + (dv[i][col] * Wv[row][col]);
                    sum_k = sum_k + (dk[i][col] * Wk[row][col]);
                }
                dx[i][row] = dx[i][row] + sum_v + sum_k;
            }
        }

        for(int row = 0; row < d; row++)
        {
            double sum_q = 0.0;
            for(int col = 0; col < d; col++)
                sum_q = sum_q + (dq[col] * Wq[row][col]);
            dx[n - 1][row] = dx[n - 1][row] + sum_q + dx_res[row];
        }

        for(int row = 0; row < d; row++)
            for(int col = 0; col < d; col++)
                Wq[row][col] = Wq[row][col] - (lr * fc.x[n - 1][row] * dq[col]);

        for(int row = 0; row < d; row++)
        {
            for(int col = 0; col < d; col++)
            {
                double sum_wk = 0.0;
                double sum_wv = 0.0;
                for(int i = 0; i < n; i++)
                {
                    sum_wk = sum_wk + (fc.x[i][row] * dk[i][col]);
                    sum_wv = sum_wv + (fc.x[i][row] * dv[i][col]);
                }
                Wk[row][col] = Wk[row][col] - (lr * sum_wk);
                Wv[row][col] = Wv[row][col] - (lr * sum_wv);
            }
        }

        for(int i = 0; i < n; i++)
        {
            int char_row = context[i];
            for(int m = 0; m < d; m++)
            {
                C[char_row][m] = C[char_row][m] - (lr * dx[i][m]);
                P[i][m] = P[i][m] - (lr * dx[i][m]);
            }
        }
    }

    // =========================================================================
    // 2. MULTI-HEAD ATTENTION (H2 heads, each D/H2 dimension + Residual Addition)
    // =========================================================================
    public static class MultiHeadCache
    {
        public double[][] x;                // N x D
        public double[][][] all_q;          // n_heads x D x head_dim
        public double[][][] all_k;          // n_heads x N x head_dim
        public double[][][] all_v;          // n_heads x N x head_dim
        public double[][] all_a;            // n_heads x N
        public double[] head_outs;          // D
        public double[] net_in;             // D (residual addition)
        public double[] h;                  // H
        public double[] probs;              // V
        public double loss;
    }

    public static MultiHeadCache forward_multi_head(int[] context, int target_idx,
                                                    double[][] C, double[][] P,
                                                    double[][][] Wq, double[][][] Wk, double[][][] Wv,
                                                    double[][] W1, double[] b1, double[][] W2, double[] b2,
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
                mc.x[i][m] = C[char_row][m] + P[i][m];
        }

        mc.all_q = new double[n_heads][1][head_dim]; // 1 vector of length head_dim per head
        mc.all_k = new double[n_heads][n][head_dim];
        mc.all_v = new double[n_heads][n][head_dim];
        mc.all_a = new double[n_heads][n];
        mc.head_outs = new double[d];

        for(int h = 0; h < n_heads; h++)
        {
            //q for head h
            double[] q_h = new double[head_dim];
            for(int col = 0; col < head_dim; col++)
            {
                double sum = 0.0;
                for(int row = 0; row < d; row++)
                    sum = sum + (mc.x[n - 1][row] * Wq[h][row][col]);
                q_h[col] = sum;
            }
            mc.all_q[h][0] = q_h;

            //k and v for head h
            for(int i = 0; i < n; i++)
            {
                for(int col = 0; col < head_dim; col++)
                {
                    double sum_k = 0.0;
                    double sum_v = 0.0;
                    for(int row = 0; row < d; row++)
                    {
                        sum_k = sum_k + (mc.x[i][row] * Wk[h][row][col]);
                        sum_v = sum_v + (mc.x[i][row] * Wv[h][row][col]);
                    }
                    mc.all_k[h][i][col] = sum_k;
                    mc.all_v[h][i][col] = sum_v;
                }
            }

            //scores and softmax
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

            //Blend head h output into slice of head_outs
            for(int m = 0; m < head_dim; m++)
            {
                double sum = 0.0;
                for(int i = 0; i < n; i++)
                    sum = sum + (mc.all_a[h][i] * mc.all_v[h][i][m]);
                mc.head_outs[h * head_dim + m] = sum;
            }
        }

        //Residual addition: head_outs + x[n-1]
        mc.net_in = new double[d];
        for(int m = 0; m < d; m++)
            mc.net_in[m] = mc.head_outs[m] + mc.x[n - 1][m];

        //Hidden layer
        mc.h = new double[h_size];
        for(int col = 0; col < h_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < d; row++)
                sum = sum + (mc.net_in[row] * W1[row][col]);
            mc.h[col] = Math.tanh(sum + b1[col]);
        }

        //Output layer
        double[] out_scores = new double[v_size];
        double max_out = -Double.MAX_VALUE;
        for(int col = 0; col < v_size; col++)
        {
            double sum = 0.0;
            for(int row = 0; row < h_size; row++)
                sum = sum + (mc.h[row] * W2[row][col]);
            out_scores[col] = sum + b2[col];
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

        mc.loss = -Math.log(mc.probs[target_idx]);
        return mc;
    }

    public static void backward_multi_head(MultiHeadCache mc, int[] context, int target_idx,
                                           double[][] C, double[][] P,
                                           double[][][] Wq, double[][][] Wk, double[][][] Wv,
                                           double[][] W1, double[] b1, double[][] W2, double[] b2,
                                           double lr, int n, int d, int n_heads, int h_size, int v_size)
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
                sum = sum + (W2[col][j] * g[j]);
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
                sum = sum + (W1[row][col] * dz[col]);
            d_net_in[row] = sum;
        }

        //Updates for W2, b2, W1, b1
        for(int j = 0; j < v_size; j++)
        {
            b2[j] = b2[j] - (lr * g[j]);
            for(int col = 0; col < h_size; col++)
                W2[col][j] = W2[col][j] - (lr * mc.h[col] * g[j]);
        }

        for(int col = 0; col < h_size; col++)
        {
            b1[col] = b1[col] - (lr * dz[col]);
            for(int row = 0; row < d; row++)
                W1[row][col] = W1[row][col] - (lr * mc.net_in[row] * dz[col]);
        }

        //d_net_in splits into d_head_outs and residual addition to x[n-1]
        double[] d_head_outs = d_net_in;
        double[] dx_res = d_net_in;

        double[][] dx = new double[n][d];
        for(int m = 0; m < d; m++)
            dx[n - 1][m] = dx[n - 1][m] + dx_res[m];

        for(int h = 0; h < n_heads; h++)
        {
            int offset = h * head_dim;
            double[] d_out_h = new double[head_dim];
            for(int m = 0; m < head_dim; m++)
                d_out_h[m] = d_head_outs[offset + m];

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

            //Accumulate dx from head h
            for(int i = 0; i < n; i++)
            {
                for(int row = 0; row < d; row++)
                {
                    double sum_v = 0.0;
                    double sum_k = 0.0;
                    for(int col = 0; col < head_dim; col++)
                    {
                        sum_v = sum_v + (dv_h[i][col] * Wv[h][row][col]);
                        sum_k = sum_k + (dk_h[i][col] * Wk[h][row][col]);
                    }
                    dx[i][row] = dx[i][row] + sum_v + sum_k;
                }
            }

            for(int row = 0; row < d; row++)
            {
                double sum_q = 0.0;
                for(int col = 0; col < head_dim; col++)
                    sum_q = sum_q + (dq_h[col] * Wq[h][row][col]);
                dx[n - 1][row] = dx[n - 1][row] + sum_q;
            }

            //Update head h weights Wq, Wk, Wv
            for(int row = 0; row < d; row++)
            {
                for(int col = 0; col < head_dim; col++)
                {
                    Wq[h][row][col] = Wq[h][row][col] - (lr * mc.x[n - 1][row] * dq_h[col]);
                    double sum_wk = 0.0;
                    double sum_wv = 0.0;
                    for(int i = 0; i < n; i++)
                    {
                        sum_wk = sum_wk + (mc.x[i][row] * dk_h[i][col]);
                        sum_wv = sum_wv + (mc.x[i][row] * dv_h[i][col]);
                    }
                    Wk[h][row][col] = Wk[h][row][col] - (lr * sum_wk);
                    Wv[h][row][col] = Wv[h][row][col] - (lr * sum_wv);
                }
            }
        }

        //Update C and P
        for(int i = 0; i < n; i++)
        {
            int char_row = context[i];
            for(int m = 0; m < d; m++)
            {
                C[char_row][m] = C[char_row][m] - (lr * dx[i][m]);
                P[i][m] = P[i][m] - (lr * dx[i][m]);
            }
        }
    }
}
