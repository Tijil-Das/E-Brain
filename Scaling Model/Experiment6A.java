public class Experiment6A
{
    static String raw_text = Data.chunk1;
    static char[] char_present_in_text;
    static int v;
    static int n = 4;
    static int d = 10;
    static int h = 100;
    static int input_dim = n * d;

    public static void main(String[] args)
    {
        Library.fill_the_characters_present_in_text(raw_text);
        char_present_in_text = Library.ch_present;
        v = char_present_in_text.length;

        System.out.println("===================================================================");
        System.out.println("                 STAGE 6A: FIXING OVERFITTING                      ");
        System.out.println("===================================================================\n");

        //1. BASELINE (from Stage 5)
        System.out.println(">>> 0. BASELINE (Stage 5 Configuration, 15 Epochs, LR=0.01, No Decay)");
        run_experiment("Baseline", raw_text, 15, 0.01, 1.0, 0.0, false);

        //2. CHANGE 1: MORE TEXT (Doubled text: 2x)
        System.out.println("\n>>> 1. CHANGE: MORE TEXT (2x data size)");
        String doubled_text = raw_text + raw_text;
        run_experiment("More Text (2x)", doubled_text, 15, 0.01, 1.0, 0.0, false);

        //3. CHANGE 2: BEST-EPOCH SAVING
        System.out.println("\n>>> 2. CHANGE: BEST-EPOCH SAVING (Track & Restore Best Val Weights)");
        run_experiment("Best-Epoch Saving", raw_text, 15, 0.01, 1.0, 0.0, true);

        //4. CHANGE 3: WEIGHT DECAY (decay factor 0.0001 per update)
        System.out.println("\n>>> 3. CHANGE: WEIGHT DECAY (0.9999 multiplier / step)");
        run_experiment("Weight Decay", raw_text, 15, 0.01, 1.0, 0.0001, false);

        //5. CHANGE 4: LEARNING RATE DECAY (Start 0.015, multiply by 0.9 / epoch)
        System.out.println("\n>>> 4. CHANGE: LEARNING RATE DECAY (Start 0.015, Decay x0.9 / epoch)");
        run_experiment("LR Decay", raw_text, 15, 0.015, 0.90, 0.0, false);
    }

    public static void run_experiment(String name, String dataset, int epochs, double start_lr, double lr_decay_factor, double weight_decay, boolean save_best)
    {
        int split_idx = (int)(dataset.length() * 0.90);
        String train_set = dataset.substring(0, split_idx);
        String val_set = dataset.substring(split_idx);

        double[][] C = Library.make_embedding_table(v, d);
        double[][] W1 = Library.make_W1_matrix(input_dim, h);
        double[] b1 = Library.make_b1_vector(h);
        double[][] W2 = Library.make_W2_matrix(h, v);
        double[] b2 = Library.make_b2_vector(v);

        double best_val = Double.MAX_VALUE;
        int best_ep = 1;
        double current_lr = start_lr;
        double final_train = 0.0;
        double final_val = 0.0;

        for(int ep = 1; ep <= epochs; ep++)
        {
            double tr_loss = Library.train_one_epoch(C, W1, b1, W2, b2, train_set, char_present_in_text, current_lr, n, d, weight_decay);
            double val_loss = Library.compute_loss_on_text(C, W1, b1, W2, b2, val_set, char_present_in_text, n, d);

            if(val_loss < best_val)
            {
                best_val = val_loss;
                best_ep = ep;
            }

            final_train = tr_loss;
            final_val = val_loss;

            current_lr = current_lr * lr_decay_factor;
        }

        System.out.printf("[%s] Final Train Loss: %.4f | Final Val Loss: %.4f | Gap: %.4f | Best Val Loss: %.4f (Epoch %d)%n",
            name, final_train, final_val, (final_val - final_train), best_val, best_ep);
    }
}
