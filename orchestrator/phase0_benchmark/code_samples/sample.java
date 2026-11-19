package io.example.util;
import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;

public final class Checksum {
    private Checksum() {}

    public static String sha256(Path file) throws IOException, NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(file)) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) != -1) md.update(buf, 0, n);
        }
        byte[] digest = md.digest();
        StringBuilder sb = new StringBuilder(64);
        for (byte b : digest) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    public static Map<String, String> checksumDir(Path dir) throws Exception {
        var result = new LinkedHashMap<String, String>();
        try (var stream = Files.walk(dir)) {
            stream.filter(Files::isRegularFile).sorted()
                  .forEach(p -> {
                      try { result.put(dir.relativize(p).toString(), sha256(p)); }
                      catch (Exception e) { throw new RuntimeException(e); }
                  });
        }
        return result;
    }
}
