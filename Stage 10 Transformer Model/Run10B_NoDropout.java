import java.util.Locale;

public class Run10B_NoDropout 
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

        TransformerLibrary.fill_the_characters_present_in_text(raw_text);
        char[] vocab = TransformerLibrary.ch_present;
        int V = vocab.length;

        // 2x text
        String full_text = raw_text + raw_text;
        int split_idx = (int)(full_text.length() * 0.90);
        String train_text = full_text.substring(0, split_idx);
        String val_text = full_text.substring(split_idx);

        System.out.println("==========================================================================");
        System.out.println("STAGE 10B: FAIR 30-EPOCH RUN (2 BLOCKS, ZERO DROPOUT, LR DECAY x0.95)");
        System.out.println("Vocabulary size: " + V);
        System.out.println("Train characters: " + train_text.length() + ", Val characters: " + val_text.length());
        System.out.println("==========================================================================\n");

        int T = 16;
        int D = 32;
        int n_heads = 4;
        int H_mlp = 100;
        int L = 2; // 2 stacked blocks
        int epochs = 30;
        double start_lr = 0.020;
        double lr_decay = 0.95;
        int stride = 8; // stride 8 with epoch jitter (307,520 token updates / epoch)

        TransformerLibrary.TransformerModel model = TransformerLibrary.TransformerModel.init(V, T, D, n_heads, H_mlp, L, 0.05);
        TransformerLibrary.TransformerModel best_model = model.copy();

        double best_val_half = Double.MAX_VALUE;
        double best_val_all = Double.MAX_VALUE;
        double train_at_best = Double.MAX_VALUE;
        int best_epoch = 1;

        int limit = train_text.length() - T - 1;
        int[] window = new int[T];
        int[] target = new int[T];
        double current_lr = start_lr;

        for(int epoch = 1; epoch <= epochs; epoch++)
        {
            long t0 = System.currentTimeMillis();
            double train_loss_sum = 0.0;
            int n_windows = 0;

            // Jitter offset so tokens rotate across different positions in the window
            int offset = (epoch * 3) % stride;

            for(int t = offset; t <= limit; t += stride)
            {
                for(int i = 0; i < T; i++)
                {
                    window[i] = TransformerLibrary.find_ch_number(train_text.charAt(t + i), vocab);
                    target[i] = TransformerLibrary.find_ch_number(train_text.charAt(t + i + 1), vocab);
                }
                // Zero dropout during training
                TransformerLibrary.ForwardCache fc = TransformerLibrary.forward(window, target, model, false, 0.0);
                TransformerLibrary.backward(fc, window, target, model, current_lr);
                train_loss_sum += fc.total_loss;
                n_windows++;
            }

            double avg_train = train_loss_sum / n_windows;
            // Eval without dropout (stride 8 for thorough validation)
            double[] val_scores = TransformerLibrary.evaluate_val_loss(val_text, vocab, model, 8);
            double val_all = val_scores[0];
            double val_half = val_scores[1];
            long elapsed = System.currentTimeMillis() - t0;

            boolean is_best = false;
            if(val_half < best_val_half)
            {
                best_val_half = val_half;
                best_val_all = val_all;
                train_at_best = avg_train;
                best_epoch = epoch;
                best_model = model.copy();
                is_best = true;
            }

            System.out.printf("Epoch %2d/%2d | Train: %.4f | Val (All 0..15): %.4f | Val (Pos 8..15): %.4f | LR: %.5f | %4d ms %s\n",
                    epoch, epochs, avg_train, val_all, val_half, current_lr, elapsed, (is_best ? "<- [BEST]" : ""));

            current_lr *= lr_decay;
        }

        System.out.println("\n--------------------------------------------------------------------------");
        System.out.printf("Run-1 Complete! Best Epoch: %d\n", best_epoch);
        System.out.printf("Best Val Loss (Positions 8..15) : %.4f\n", best_val_half);
        System.out.printf("Best Val Loss (All Pos 0..15)   : %.4f\n", best_val_all);
        System.out.printf("Train Loss at Best Epoch        : %.4f\n", train_at_best);
        System.out.printf("Comparison to Stage 9B (1.7511) : %s (Diff: %+.4f)\n",
                (best_val_half < 1.7511 ? "BEATS STAGE 9B" : "TRAILS STAGE 9B"), (best_val_half - 1.7511));
        System.out.println("--------------------------------------------------------------------------\n");

        // Save best model to disk
        String save_path = "best_model_10b.bin";
        try
        {
            TransformerLibrary.save_model(best_model, save_path);
            System.out.println("Saved best model weights to: " + save_path);

            // Test loading model
            TransformerLibrary.TransformerModel loaded = TransformerLibrary.load_model(save_path);
            System.out.println("Successfully loaded model back from disk!");

            // Generate 300 characters at temperature 0.8
            System.out.println("\n--- 300 Generated Characters from Best Model at Temperature 0.8 ---");
            String seed = "Arthur Pendelton";
            String sample = TransformerLibrary.generate_text(loaded, seed, 300, vocab, 0.8);
            System.out.println(seed + sample);
            System.out.println("--------------------------------------------------------------------------");
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}
