public class BenchmarkFast 
{
    public static void main(String[] args) 
    {
        String raw_text = Data.chunk1;
        Library.fill_the_characters_present_in_text(raw_text);
        char[] vocab = Library.ch_present;
        int V = vocab.length;
        String full_text = raw_text + raw_text;
        int split_idx = (int)(full_text.length() * 0.90);
        String train_text = full_text.substring(0, split_idx);

        int N = 16, D = 32, n_heads = 4, H = 100;
        Library.ModelWeights mw = Library.init_model(V, N, D, n_heads, H, 0.05, 0.05);
        FastEngine.Workspace ws = new FastEngine.Workspace(N, D, n_heads, H, V);

        int train_len = train_text.length() - N;
        int[] ctx = new int[N];

        long t0 = System.currentTimeMillis();
        double sum_loss = 0.0;
        for(int t = 0; t < train_len; t++)
        {
            for(int i = 0; i < N; i++)
                ctx[i] = Library.find_ch_number(train_text.charAt(t + i), vocab);
            int target = Library.find_ch_number(train_text.charAt(t + N), vocab);

            FastEngine.forward_fast(ctx, target, mw, ws, N, D, n_heads, H, V);
            FastEngine.backward_fast(ctx, target, mw, ws, 0.015, 0.015, N, D, n_heads, H, V);
            sum_loss += ws.loss;
        }
        long t1 = System.currentTimeMillis();
        System.out.printf("1 Epoch of %d steps took: %d ms (%.2f seconds)! Train loss: %.4f\n",
                train_len, (t1 - t0), (t1 - t0) / 1000.0, (sum_loss / train_len));
    }
}
