import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class ExperimentTracker 
{
    public String runId;
    public String purpose;
    public Map<String, String> config;
    public Dataset dataset;
    public String gitCommit;
    public String runDir;
    public File metricsCsv;
    public PrintWriter csvWriter;

    public double bestValLoss = Double.POSITIVE_INFINITY;
    public int bestEpoch = -1;
    public double trainLossAtBest = 0.0;
    public int totalParams = 0;

    public static ExperimentTracker init(String purpose, Map<String, String> config, Dataset dataset) 
    {
        ExperimentTracker tracker = new ExperimentTracker();
        tracker.purpose = purpose;
        tracker.config = new LinkedHashMap<>(config);
        tracker.dataset = dataset;

        // Generate Run ID
        String dateStr = new SimpleDateFormat("yyyyMMdd").format(new Date());
        File baseDir = new File("runs");
        if (!baseDir.exists()) baseDir.mkdirs();

        int counter = 1;
        while (new File(baseDir, String.format("RUN_%s_%03d", dateStr, counter)).exists()) 
        {
            counter++;
        }
        tracker.runId = String.format("RUN_%s_%03d", dateStr, counter);
        tracker.runDir = "runs/" + tracker.runId;
        new File(tracker.runDir).mkdirs();

        // Retrieve git commit hash
        tracker.gitCommit = getGitCommitHash();

        // Initialize CSV logger
        tracker.metricsCsv = new File(tracker.runDir, tracker.runId + "_metrics.csv");
        try 
        {
            tracker.csvWriter = new PrintWriter(new BufferedWriter(new FileWriter(tracker.metricsCsv)));
            tracker.csvWriter.println("epoch,train_loss,val_loss_all,val_loss_pos8_15,learning_rate,time_ms");
            tracker.csvWriter.flush();
        } 
        catch (IOException e) 
        {
            System.err.println("Failed to initialize CSV logger: " + e.getMessage());
        }

        // Print header
        System.out.println("==========================================================================");
        System.out.printf("TRACKED RUN: %s | Purpose: %s\n", tracker.runId, tracker.purpose);
        System.out.printf("Git Commit: %s | Data Hash: %s\n", tracker.gitCommit, dataset.sha256Checksum.substring(0, 16) + "...");
        System.out.printf("Dataset: %s (%d characters) | Vocab: %d tokens\n", dataset.filename, dataset.characterCount, dataset.vocab.size());
        System.out.printf("Config: %s\n", tracker.config.toString());
        System.out.println("==========================================================================");

        return tracker;
    }

    public void logEpoch(int epoch, double trainLoss, double valLossAll, double valLossPos8_15, double lr, long timeMs, Transformer model) 
    {
        if (csvWriter != null) 
        {
            csvWriter.printf("%d,%.6f,%.6f,%.6f,%.6f,%d\n", epoch, trainLoss, valLossAll, valLossPos8_15, lr, timeMs);
            csvWriter.flush();
        }

        boolean isBest = valLossPos8_15 < bestValLoss;
        if (isBest) 
        {
            bestValLoss = valLossPos8_15;
            bestEpoch = epoch;
            trainLossAtBest = trainLoss;

            // Save best checkpoint
            try 
            {
                String bestPath = runDir + "/" + runId + "_best.bin";
                model.saveCheckpoint(bestPath);
            } 
            catch (IOException e) 
            {
                System.err.println("Error saving best checkpoint: " + e.getMessage());
            }
        }

        // Checkpoint every 5 epochs
        if (epoch % 5 == 0) 
        {
            try 
            {
                String epochPath = String.format("%s/%s_epoch_%03d.bin", runDir, runId, epoch);
                model.saveCheckpoint(epochPath);
            } 
            catch (IOException e) 
            {
                System.err.println("Error saving epoch checkpoint: " + e.getMessage());
            }
        }

        System.out.printf("Epoch %2d | Train: %.4f | Val (All): %.4f | Val (8..15): %.4f | LR: %.5f | %d ms %s\n",
                epoch, trainLoss, valLossAll, valLossPos8_15, lr, timeMs, isBest ? "<- [BEST]" : "");
    }

    public void finishRun(Transformer model, String generatedSample) 
    {
        if (csvWriter != null) 
        {
            csvWriter.close();
        }

        totalParams = model.totalParameters();

        // 1. Append to master_experiments.csv
        File masterCsv = new File("runs/master_experiments.csv");
        boolean exists = masterCsv.exists();
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(masterCsv, true)))) 
        {
            if (!exists) 
            {
                pw.println("run_id,timestamp,purpose,git_commit,data_file,data_hash,data_chars,vocab_size,N,D,H,L,H_mlp,params,optimizer,batch_size,lr_init,lr_decay,dropout,epochs,best_epoch,train_loss_best,val_loss_best");
            }
            pw.printf("%s,%s,\"%s\",%s,%s,%s,%d,%d,%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%d,%.4f,%.4f\n",
                    runId,
                    new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()),
                    purpose,
                    gitCommit,
                    dataset.filename,
                    dataset.sha256Checksum.substring(0, 16),
                    dataset.characterCount,
                    dataset.vocab.size(),
                    config.getOrDefault("N", "16"),
                    config.getOrDefault("D", "32"),
                    config.getOrDefault("H", "4"),
                    config.getOrDefault("L", "2"),
                    config.getOrDefault("H_mlp", "100"),
                    totalParams,
                    config.getOrDefault("optimizer", "Adam"),
                    config.getOrDefault("batch_size", "16"),
                    config.getOrDefault("lr", "0.001"),
                    config.getOrDefault("decay", "0.95"),
                    config.getOrDefault("dropout", "0.0"),
                    config.getOrDefault("epochs", "30"),
                    bestEpoch,
                    trainLossAtBest,
                    bestValLoss
            );
        } 
        catch (IOException e) 
        {
            System.err.println("Failed to write to master_experiments.csv: " + e.getMessage());
        }

        // 2. Append markdown report to runs/master_log.md
        File masterMd = new File("runs/master_log.md");
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(masterMd, true)))) 
        {
            pw.printf("### Run `%s`: %s\n\n", runId, purpose);
            pw.printf("- **Date/Time:** %s\n", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            pw.printf("- **Git Commit:** `%s`\n", gitCommit);
            pw.printf("- **Data Source:** `%s` (%d chars, SHA-256: `%s`)\n", dataset.filename, dataset.characterCount, dataset.sha256Checksum.substring(0, 16));
            pw.printf("- **Model Architecture:** $L=%s, H=%s, D=%s, H_{mlp}=%s$, Params: %d\n",
                    config.get("L"), config.get("H"), config.get("D"), config.get("H_mlp"), totalParams);
            pw.printf("- **Optimization:** %s, LR=%s, Batch=%s, Dropout=%s\n",
                    config.get("optimizer"), config.get("lr"), config.get("batch_size"), config.get("dropout"));
            pw.printf("- **Outcome:** Best Epoch **%d** | Train Loss: **%.4f** | Best Val Loss (Pos 8..15): **%.4f**\n\n",
                    bestEpoch, trainLossAtBest, bestValLoss);
            pw.println("```text");
            pw.println(generatedSample.trim());
            pw.println("```\n\n---\n");
        } 
        catch (IOException e) 
        {
            System.err.println("Failed to write to master_log.md: " + e.getMessage());
        }

        System.out.println("==========================================================================");
        System.out.printf("RUN %s COMPLETE! Best Epoch: %d | Val Loss: %.4f | Train Loss: %.4f\n",
                runId, bestEpoch, bestValLoss, trainLossAtBest);
        System.out.println("==========================================================================\n");
    }

    private static String getGitCommitHash() 
    {
        try 
        {
            Process p = Runtime.getRuntime().exec("git rev-parse HEAD");
            BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line = reader.readLine();
            if (line != null && !line.trim().isEmpty()) 
            {
                return line.trim();
            }
        } 
        catch (Exception ignored) {}
        return "UNKNOWN_COMMIT";
    }
}
