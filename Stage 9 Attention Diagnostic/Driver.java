import java.util.Locale;

public class Driver 
{
    public static void main(String[] args) 
    {
        Locale.setDefault(Locale.US);

        String raw_text = Data.chunk1;
        if(raw_text == null || raw_text.length() == 0)
        {
            System.out.println("Error: Text could not be loaded.");
            return;
        }

        Library.fill_the_characters_present_in_text(raw_text);
        char[] vocab = Library.ch_present;
        int V = vocab.length;

        // 2x text as in Stage 6A
        String full_text = raw_text + raw_text;
        int split_idx = (int)(full_text.length() * 0.90);
        String train_text = full_text.substring(0, split_idx);
        String val_text = full_text.substring(split_idx);

        System.out.println("=================================================");
        System.out.println("STAGE 9: FAIR LONG RUN & ATTENTION DIAGNOSTICS");
        System.out.println("Vocabulary size: " + V);
        System.out.println("Train characters: " + train_text.length() + ", Val characters: " + val_text.length());
        System.out.println("=================================================\n");

        int N = 16;
        int D = 32;
        int n_heads = 4;
        int head_dim = D / n_heads; // 8
        int H = 100;
        int epochs = 30;
        double start_lr = 0.015;
        double lr_decay = 0.95;

        // -------------------------------------------------------------
        // 9B: A FAIR LONG RUN (30 Epochs, Best-Epoch Saving)
        // -------------------------------------------------------------
        System.out.println("--- 9B: Fair Long Run (30 Epochs, 2x Text, LR Decay, Best-Epoch Saving) ---");
        System.out.println("Configuration: N=16, D=32, H_heads=4 (dk=8), MLP_H=100, Residual=Additive");
        System.out.printf("Start LR: %.4f | LR Decay: %.2f / epoch | Baseline to beat: ~1.86\n\n", start_lr, lr_decay);

        // Initialize model with standard scale 0.05
        Library.ModelWeights model = Library.init_model(V, N, D, n_heads, H, 0.05, 0.05);
        Library.ModelWeights best_model = model.copy();

        double best_val_loss = Double.MAX_VALUE;
        int best_epoch = 1;

        int train_len = train_text.length() - N;
        int val_len = val_text.length() - N;

        double current_lr = start_lr;
        int[] ctx = new int[N];

        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            long t0 = System.currentTimeMillis();
            double train_loss_sum = 0.0;

            for(int t = 0; t < train_len; t++)
            {
                for(int i = 0; i < N; i++)
                    ctx[i] = Library.find_ch_number(train_text.charAt(t + i), vocab);
                int target = Library.find_ch_number(train_text.charAt(t + N), vocab);

                Library.MultiHeadCache mc = Library.forward_multi_head(ctx, target, model, N, D, n_heads, H, V);
                Library.backward_multi_head(mc, ctx, target, model, current_lr, current_lr, N, D, n_heads, H, V);
                train_loss_sum += mc.loss;
            }

            double avg_train = train_loss_sum / train_len;
            double avg_val = Library.compute_val_loss(val_text, vocab, model, N, D, n_heads, H, V);
            long elapsed = System.currentTimeMillis() - t0;

            boolean is_best = false;
            if(avg_val < best_val_loss)
            {
                best_val_loss = avg_val;
                best_epoch = epoch;
                best_model = model.copy();
                is_best = true;
            }

            System.out.printf("Epoch %2d/%2d | Train: %.4f | Val: %.4f | Gap: %+.4f | LR: %.5f | %4d ms %s\n",
                    epoch, epochs, avg_train, avg_val, (avg_val - avg_train), current_lr, elapsed, (is_best ? "<- [BEST]" : ""));

            current_lr *= lr_decay;
        }

        System.out.println("\n-------------------------------------------------");
        System.out.printf("Long Run Finished! Best Val Loss: %.4f at Epoch %d\n", best_val_loss, best_epoch);
        System.out.printf("Comparison to non-attention baseline (~1.86): %s (Diff: %+.4f)\n",
                (best_val_loss < 1.86 ? "BEATS BASELINE" : "TRAILS BASELINE"), (best_val_loss - 1.86));
        System.out.println("-------------------------------------------------\n");

        // -------------------------------------------------------------
        // 9A: DIAGNOSTIC - ATTENTION WEIGHTS INSPECTION FOR A CONTEXT WINDOW
        // -------------------------------------------------------------
        System.out.println("--- 9A: Attention Weights Diagnostic on Best Model ---");
        // Take a window from validation set
        int sample_pos = 100;
        String sample_window = val_text.substring(sample_pos, sample_pos + N);
        char next_char = val_text.charAt(sample_pos + N);
        print_window_attention(best_model, sample_window, next_char, vocab, N, D, n_heads, H, V);

        // Take another sample window (around spaces and punctuation)
        int sample_pos2 = 250;
        String sample_window2 = val_text.substring(sample_pos2, sample_pos2 + N);
        char next_char2 = val_text.charAt(sample_pos2 + N);
        System.out.println();
        print_window_attention(best_model, sample_window2, next_char2, vocab, N, D, n_heads, H, V);
    }

    public static void print_window_attention(Library.ModelWeights mw, String window, char target_char,
                                             char[] vocab, int N, int D, int n_heads, int H, int V)
    {
        int[] ctx = new int[N];
        for(int i = 0; i < N; i++)
            ctx[i] = Library.find_ch_number(window.charAt(i), vocab);
        int target = Library.find_ch_number(target_char, vocab);

        Library.MultiHeadCache mc = Library.forward_multi_head(ctx, target, mw, N, D, n_heads, H, V);

        System.out.println("Context Window (length 16): \"" + window + "\" -> Next Char: '" + target_char + "'");
        System.out.println("Uniform Baseline Weight (1/16) = 0.0625");
        System.out.println("Position index :  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15");
        System.out.print("Characters     : ");
        for(int i = 0; i < N; i++)
        {
            char c = window.charAt(i);
            String display = (c == ' ') ? " " : String.valueOf(c);
            System.out.printf(" %2s ", display);
        }
        System.out.println();

        for(int h = 0; h < n_heads; h++)
        {
            System.out.printf("Head %d Weights  : ", h);
            int max_idx = 0;
            double max_w = -1.0;
            for(int i = 0; i < N; i++)
            {
                double w = mc.all_a[h][i];
                System.out.printf("%.2f ", w);
                if(w > max_w)
                {
                    max_w = w;
                    max_idx = i;
                }
            }
            char max_char = window.charAt(max_idx);
            String char_desc = (max_char == ' ') ? "'space'" : "'" + max_char + "'";
            System.out.printf(" | Max: %.3f at pos %2d (%s)\n", max_w, max_idx, char_desc);
        }
    }
}
