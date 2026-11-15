import java.io.*;

public class ExportParityRef 
{
    public static void main(String[] args) throws Exception 
    {
        int V = 68, T = 16, D = 32, nHeads = 4, H_mlp = 64, L = 2;
        Transformer model = Transformer.init(V, T, D, nHeads, H_mlp, L, true, 0.05, 12345);

        // Export to binary file
        String path = "parity_ref.bin";
        model.saveCheckpoint(path);
        System.out.println("Exported parity reference model to: " + path);

        // Also run forward pass on a fixed test sequence and print output logits
        int[] testTokens = {1, 5, 12, 24, 3, 19, 8, 33, 42, 10, 15, 2, 7, 28, 4, 11};
        int[] testTargets = {5, 12, 24, 3, 19, 8, 33, 42, 10, 15, 2, 7, 28, 4, 11, 1};

        Transformer.Cache cache = model.forward(testTokens, testTargets, false, 0.0);
        System.out.printf("Java Reference Loss: %.8f\n", cache.avgLoss);

        // Save reference logits to file for exact byte/float comparison
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream("java_logits.bin"))) 
        {
            dos.writeDouble(cache.avgLoss);
            for (int t = 0; t < T; t++) 
            {
                for (int v = 0; v < V; v++) 
                {
                    dos.writeDouble(cache.logits[t][v]);
                }
            }
        }
        System.out.println("Saved Java reference logits to java_logits.bin");
    }
}
