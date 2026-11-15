import java.io.File;
import java.util.*;

public class Phase3AblationRunner 
{
    public static void main(String[] args) throws Exception 
    {
        // 1. Load Dataset
        String dataPath = "../Stage 10 Transformer Model/Text.txt";
        if (!new File(dataPath).exists()) 
        {
            dataPath = "Text.txt";
        }
        Dataset dataset = Dataset.loadAndPrepare(dataPath, null, 0.85, 0.10, 0.05);

        // 2. Run Phase 2 Correctness Suite
        boolean testsPassed = CorrectnessSuite.runAllTests(dataset.vocab);
        if (!testsPassed) 
        {
            System.err.println("Pre-flight correctness checks failed. Halting.");
            System.exit(1);
        }

        System.out.println("==========================================================================");
        System.out.println("PHASE 3: COMPONENT ABLATION SUITE ON CONTROLLED SMALL DATA");
        System.out.println("Comparing: Baseline SGD -> Adam -> Mini-Batches -> Warmup -> LayerNorm");
        System.out.println("==========================================================================\n");

        int epochs = 4; // Fast controlled comparison on identical data subset
        int T = 16, D = 32, nHeads = 4, H_mlp = 64, L = 2;

        // Run 1: Baseline SGD (Online, No LN, No Warmup)
        runExperiment("Baseline: SGD Online", dataset, T, D, nHeads, H_mlp, L, "SGD", 1, false, false, 0.02, 0.95, epochs, 42);

        // Run 2: Add Adam Optimizer
        runExperiment("Ablation 1: + Adam Optimizer", dataset, T, D, nHeads, H_mlp, L, "Adam", 1, false, false, 0.003, 0.95, epochs, 42);

        // Run 3: Add Mini-Batches (Batch Size = 16)
        runExperiment("Ablation 2: + Mini-Batches (B=16)", dataset, T, D, nHeads, H_mlp, L, "Adam", 16, false, false, 0.005, 0.95, epochs, 42);

        // Run 4: Add Learning Rate Warmup
        runExperiment("Ablation 3: + LR Warmup (1 Epoch)", dataset, T, D, nHeads, H_mlp, L, "Adam", 16, true, false, 0.005, 0.95, epochs, 42);

        // Run 5: Add Layer Normalization (Pre-LN)
        runExperiment("Ablation 4: + Pre-LayerNorm", dataset, T, D, nHeads, H_mlp, L, "Adam", 16, true, true, 0.005, 0.95, epochs, 42);
    }

    public static void runExperiment(String purpose, Dataset dataset, int T, int D, int nHeads, int H_mlp, int L,
                                     String optimizer, int batchSize, boolean useWarmup, boolean useLN,
                                     double initLr, double lrDecay, int epochs, long seed) throws Exception 
    {
        Map<String, String> config = new LinkedHashMap<>();
        config.put("T", String.valueOf(T));
        config.put("D", String.valueOf(D));
        config.put("nHeads", String.valueOf(nHeads));
        config.put("H_mlp", String.valueOf(H_mlp));
        config.put("L", String.valueOf(L));
        config.put("optimizer", optimizer);
        config.put("batch_size", String.valueOf(batchSize));
        config.put("warmup", String.valueOf(useWarmup));
        config.put("layernorm", String.valueOf(useLN));
        config.put("lr", String.valueOf(initLr));
        config.put("decay", String.valueOf(lrDecay));
        config.put("epochs", String.valueOf(epochs));

        ExperimentTracker tracker = ExperimentTracker.init(purpose, config, dataset);
        Transformer model = Transformer.init(dataset.vocab.size(), T, D, nHeads, H_mlp, L, useLN, 0.02, seed);

        // Slice dataset for uniform comparison
        int maxChars = 15000;
        int nTrain = Math.min(dataset.trainTokens.length, maxChars);
        int stride = 8;
        int numWindows = (nTrain - T) / stride;

        int warmupEpochs = useWarmup ? 1 : 0;
        double currentLr = useWarmup ? (initLr / 10.0) : initLr;

        for (int ep = 1; ep <= epochs; ep++) 
        {
            long t0 = System.currentTimeMillis();

            // LR schedule: Warmup over first 2 epochs, then exponential decay
            if (useWarmup) 
            {
                if (ep <= warmupEpochs) 
                {
                    currentLr = initLr * ((double) ep / warmupEpochs);
                } 
                else 
                {
                    currentLr = initLr * Math.pow(lrDecay, ep - warmupEpochs);
                }
            } 
            else 
            {
                currentLr = initLr * Math.pow(lrDecay, ep - 1);
            }

            // Training loop with mini-batch accumulation
            double totalTrainLoss = 0.0;
            int trainSteps = 0;

            Transformer.Gradients batchAccum = new Transformer.Gradients(model.V, T, D, nHeads, model.dk, H_mlp, L);
            int inBatch = 0;

            for (int w = 0; w < numWindows; w++) 
            {
                int start = w * stride;
                int[] tok = new int[T];
                int[] tgt = new int[T];
                for (int i = 0; i < T; i++) 
                {
                    tok[i] = dataset.trainTokens[start + i];
                    tgt[i] = dataset.trainTokens[start + i + 1];
                }

                Transformer.Cache cache = model.forward(tok, tgt, true, 0.0);
                totalTrainLoss += cache.avgLoss;
                trainSteps++;

                Transformer.Gradients g = model.backward(cache);
                batchAccum.add(g);
                inBatch++;

                if (inBatch >= batchSize || w == numWindows - 1) 
                {
                    batchAccum.scale(1.0 / inBatch);
                    if ("Adam".equalsIgnoreCase(optimizer)) 
                    {
                        model.stepAdam(batchAccum, currentLr, 0.9, 0.999, 1e-8, 1e-4);
                    } 
                    else 
                    {
                        model.stepSGD(batchAccum, currentLr, 1e-4);
                    }
                    batchAccum.zero();
                    inBatch = 0;
                }
            }

            double avgTrain = totalTrainLoss / trainSteps;

            // Validation evaluation
            double valSumAll = 0.0;
            double valSumPos8_15 = 0.0;
            int valWindows = (dataset.valTokens.length - T) / stride;

            for (int w = 0; w < valWindows; w++) 
            {
                int start = w * stride;
                int[] tok = new int[T];
                int[] tgt = new int[T];
                for (int i = 0; i < T; i++) 
                {
                    tok[i] = dataset.valTokens[start + i];
                    tgt[i] = dataset.valTokens[start + i + 1];
                }

                Transformer.Cache c = model.forward(tok, tgt, false, 0.0);
                valSumAll += c.avgLoss;
                double posSum = 0.0;
                for (int i = 8; i < T; i++) posSum += c.losses[i];
                valSumPos8_15 += posSum / 8.0;
            }

            double avgValAll = valSumAll / valWindows;
            double avgValPos8_15 = valSumPos8_15 / valWindows;
            long t1 = System.currentTimeMillis();

            tracker.logEpoch(ep, avgTrain, avgValAll, avgValPos8_15, currentLr, t1 - t0, model);
        }

        String sample = model.generate(dataset.vocab, "Arthur Pendelton ", 200, 0.8);
        tracker.finishRun(model, sample);
    }
}
