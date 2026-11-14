import java.util.Locale;

public class FastDriver30 
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
        System.out.println("STAGE 10B: FAIR 30-EPOCH RUN (2 BLOCKS, DROPOUT p=0.10, LR DECAY x0.95)");
        System.out.println("Vocabulary size: " + V);
        System.out.println("Train characters: " + train_text.length() + ", Val characters: " + val_text.length());
        System.out.println("==========================================================================\n");

        int T = 16;
        int D = 32;
        int n_heads = 4;
        int H_mlp = 100;
        int L = 2; // 2 stacked blocks
        int epochs = 30;
        double start_lr = 0.015;
        double lr_decay = 0.95;
        double dropout_p = 0.10;
        int stride = 16; // non-overlapping windows (153,760 tokens / epoch)

        TransformerLibrary.TransformerModel model = TransformerLibrary.TransformerModel.init(V, T, D, n_heads, H_mlp, L, 0.05);
        TransformerLibrary.TransformerModel best_model = model.copy();

        double best_val_half = Double.MAX_VALUE;
        double best_val_all = Double.MAX_VALUE;
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

            for(int t = 0; t <= limit; t += stride)
            {
                for(int i = 0; i < T; i++)
                {
                    window[i] = TransformerLibrary.find_ch_number(train_text.charAt(t + i), vocab);
                    target[i] = TransformerLibrary.find_ch_number(train_text.charAt(t + i + 1), vocab);
                }
                // Forward with dropout p=0.10
                TransformerLibrary.ForwardCache fc = TransformerLibrary.forward(window, target, model, true, dropout_p);
                TransformerLibrary.backward(fc, window, target, model, current_lr);
                train_loss_sum += fc.total_loss;
                n_windows++;
            }

            double avg_train = train_loss_sum / n_windows;
            // Eval without dropout (stride 16 for val)
            double[] val_scores = TransformerLibrary.evaluate_val_loss(val_text, vocab, model, 16);
            double val_all = val_scores[0];
            double val_half = val_scores[1];
            long elapsed = System.currentTimeMillis() - t0;

            boolean is_best = false;
            if(val_half < best_val_half)
            {
                best_val_half = val_half;
                best_val_all = val_all;
                best_epoch = epoch;
                best_model = model.copy();
                is_best = true;
            }

            System.out.printf("Epoch %2d/%2d | Train (w/drop): %.4f | Val (All 0..15): %.4f | Val (Pos 8..15): %.4f | LR: %.5f | %4d ms %s\n",
                    epoch, epochs, avg_train, val_all, val_half, current_lr, elapsed, (is_best ? "<- [BEST]" : ""));

            current_lr *= lr_decay;
        }

        System.out.println("\n--------------------------------------------------------------------------");
        System.out.printf("30-Epoch Run Complete! Best Val (Pos 8..15): %.4f | Best Val (All Pos): %.4f at Epoch %d\n",
                best_val_half, best_val_all, best_epoch);
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
