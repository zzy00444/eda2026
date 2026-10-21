"""Losslessly pack the periodic envelope, exploiting exact modulo-12 congruence."""
import json
import zlib
import numpy as np
from displacement_assets import sha
from displacement_analyze import OUT
from project_paths import BUILD, REPO_ROOT


def main():
    a = np.load(BUILD / 'architecture_table.npy').astype(np.int32)
    meta = json.loads((BUILD / 'architecture_meta.json').read_text())
    r = np.arange(120)
    q = []
    for d, (ax, by) in enumerate(meta['slopes_scaled12']):
        values = a[d]+(ax*(r//12)+by*(r%12))[None, :, None]
        assert (values%12 == 0).all(), 'Architecture congruence must be exact'
        q.append(values//12)
    q = np.ascontiguousarray(np.stack(q).transpose(0, 1, 3, 2), dtype='<i2')
    delta = np.diff(q, axis=-1, prepend=np.zeros((*q.shape[:-1], 1), np.int16)).astype('<i2')
    assert np.array_equal(delta.astype(np.int32).cumsum(axis=-1), q)
    flat = delta.ravel()
    escape = (flat < -127) | (flat > 127)
    lengths = 1+2*escape.astype(np.int32)
    starts = np.r_[0, lengths[:-1].cumsum()]
    stream = np.empty(int(lengths.sum()), np.uint8)
    stream[starts] = np.where(escape, 128, flat).astype(np.uint8)
    stream[starts[escape]+1] = (flat[escape]&255).astype(np.uint8)
    stream[starts[escape]+2] = ((flat[escape]>>8)&255).astype(np.uint8)
    # Verify all values; runtime independently checks CRC of the reconstructed table.
    decoded = stream[starts].astype(np.int8).astype(np.int16)
    decoded[escape] = (stream[starts[escape]+1].astype(np.uint16) |
                       (stream[starts[escape]+2].astype(np.uint16)<<8)).view(np.int16)
    assert np.array_equal(decoded, flat)
    compressed = zlib.compress(stream.tobytes(), 6)
    assert zlib.decompress(compressed) == stream.tobytes()
    target = OUT / 'stage9_bound.z'
    target.write_bytes(compressed)
    runtime_table = np.ascontiguousarray(q.transpose(1, 2, 3, 0), dtype='<i2')
    report = {'codec': 'exact /12; source-target-phase order; phase delta; signed byte with 0x80+i16 escape; zlib6',
              'shape': list(q.shape), 'compressed_bytes': len(compressed), 'decoded_stream_bytes': len(stream),
              'table_bytes': runtime_table.nbytes, 'table_crc32': zlib.crc32(runtime_table.tobytes()),
              'compressed_sha256': sha(target), 'source_table_sha256': sha(BUILD / 'architecture_table.npy'),
              'source_meta_sha256': sha(BUILD / 'architecture_meta.json'),
              'slopes': meta['slopes_scaled12'], 'all_values_roundtrip_verified': True}
    (REPO_ROOT / 'docs/stage9_bound_encoding.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
