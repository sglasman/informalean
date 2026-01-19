from datasketch import MinHash, MinHashLSH
from informalean.config import DataConfig
from informalean.util.string_helpers import shingle


def minhash_shingles(s: str, data_config: DataConfig) -> MinHash:
    minhash = MinHash(num_perm=data_config.minhash_num_perm)
    for sh in shingle(s, data_config.minhash_shingle_length):
        minhash.update(sh.encode("utf8"))
    return minhash


def minhash_lsh(
    strings: list[str], data_config: DataConfig
) -> list[list[str]]:
    lsh = MinHashLSH(
        threshold=data_config.minhash_lsh_threshold,
        num_perm=data_config.minhash_num_perm,
    )
    minhashes = [minhash_shingles(s, data_config) for s in strings]
    # with Pool() as pool:
    #     minhashes = pool.starmap(
    #         minhash_shingles,
    #         [(s, data_config) for s in strings]
    #     )
    for i, minhash in enumerate(minhashes):
        lsh.insert(i, minhash)
    return [lsh.query(minhash) for minhash in minhashes]
