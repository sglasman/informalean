import faiss
import numpy as np
import logging

from informalean.config import DataConfig


logger = logging.getLogger(__name__)


def ann(data: np.array, data_config: DataConfig):
    d = data.shape[1]
    index = faiss.IndexIVFFlat(
        faiss.IndexFlatIP(d),
        d,
        data_config.faiss_statement_nlist,
        faiss.METRIC_INNER_PRODUCT,
    )
    n = data.shape[0]
    rows_to_train = data[
        np.random.choice(n, data_config.faiss_statement_n_train, replace=False)
    ]
    logger.info(f"Training IVF on {data_config.faiss_statement_n_train} vectors")
    index.train(rows_to_train)
    logger.info("Trained IVF index. Adding all vectors...")
    index.add(data)
    return index
