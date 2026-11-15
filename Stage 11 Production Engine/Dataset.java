import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.*;

public class Dataset 
{
    public String rawText;
    public String cleanedText;
    public String filename;
    public long characterCount;
    public String sha256Checksum;

    public int[] trainTokens;
    public int[] valTokens;
    public int[] testTokens;

    public Vocab vocab;

    public static Dataset loadAndPrepare(String filePath, Vocab existingVocab, double trainFrac, double valFrac, double testFrac) throws Exception 
    {
        Dataset ds = new Dataset();
        ds.filename = new File(filePath).getName();

        byte[] rawBytes = Files.readAllBytes(Paths.get(filePath));
        ds.rawText = new String(rawBytes, StandardCharsets.UTF_8);

        // Compute SHA-256 of the raw file
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] hash = digest.digest(rawBytes);
        StringBuilder hexString = new StringBuilder();
        for (byte b : hash) 
        {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) hexString.append('0');
            hexString.append(hex);
        }
        ds.sha256Checksum = hexString.toString();

        // Clean text:
        // 1. Standardize CRLF to LF
        // 2. Remove control characters (except \n, \t, and standard printable)
        // 3. Normalize multiple blank lines to at most two
        String cleaned = ds.rawText.replace("\r\n", "\n").replace("\r", "\n");
        StringBuilder sb = new StringBuilder(cleaned.length());
        for (int i = 0; i < cleaned.length(); i++) 
        {
            char c = cleaned.charAt(i);
            if (c == '\n' || c == '\t' || (c >= 32 && c <= 126) || Character.isLetterOrDigit(c)) 
            {
                sb.append(c);
            }
        }
        ds.cleanedText = sb.toString();
        ds.characterCount = ds.cleanedText.length();

        // Build or use vocab
        if (existingVocab == null) 
        {
            ds.vocab = Vocab.buildFromText(ds.cleanedText, 2);
        } 
        else 
        {
            ds.vocab = existingVocab;
        }

        // Split text sequentially
        int n = ds.cleanedText.length();
        int trainEnd = (int)(n * trainFrac);
        int valEnd = (int)(n * (trainFrac + valFrac));

        String trainPart = ds.cleanedText.substring(0, trainEnd);
        String valPart = ds.cleanedText.substring(trainEnd, valEnd);
        String testPart = ds.cleanedText.substring(valEnd);

        ds.trainTokens = ds.vocab.encodeString(trainPart);
        ds.valTokens = ds.vocab.encodeString(valPart);
        ds.testTokens = ds.vocab.encodeString(testPart);

        return ds;
    }
}
