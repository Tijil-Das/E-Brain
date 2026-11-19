package shard

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Reader reads uint16 token ids from a binary shard file.
type Reader struct {
	r   *bufio.Reader
	f   *os.File
	pos int64
}

func Open(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("shard.Open: %w", err)
	}
	return &Reader{r: bufio.NewReaderSize(f, 1<<20), f: f}, nil
}

func (r *Reader) ReadTokens(buf []uint16) (int, error) {
	for i := range buf {
		var v uint16
		if err := binary.Read(r.r, binary.LittleEndian, &v); err != nil {
			if err == io.EOF {
				return i, io.EOF
			}
			return i, fmt.Errorf("read token %d: %w", r.pos+int64(i), err)
		}
		buf[i] = v
	}
	r.pos += int64(len(buf))
	return len(buf), nil
}

func (r *Reader) Close() error { return r.f.Close() }
