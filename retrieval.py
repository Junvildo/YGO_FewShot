import numpy as np
import faiss

def _retrieve_knn_faiss_inner_product(query_embeddings, db_embeddings, k):
    """
        Retrieve k nearest neighbors based on inner product using CPU.

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    index = faiss.IndexFlatIP(db_embeddings.shape[1])  # Inner product index
    index.add(db_embeddings)
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return dists, retrieved_result_indices

def _retrieve_knn_faiss_euclidean(query_embeddings, db_embeddings, k):
    """
        Retrieve k nearest neighbors based on Euclidean distance using CPU.

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    index = faiss.IndexFlatL2(db_embeddings.shape[1])  # Euclidean distance index
    index.add(db_embeddings)
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return dists, retrieved_result_indices

def evaluate_recall_at_k(dists, results, query_labels, db_labels, k):
    """
        Evaluate Recall@k based on retrieval results.

        Args:
            dists:          numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors for each query
            results:        numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors for each query
            query_labels:   list of labels for each query
            db_labels:      list of labels for each db
            k:              number of nn results to evaluate

        Returns:
            recall_at_k:    Recall@k in percentage
    """
    self_retrieval = query_labels is db_labels
    expected_result_size = k + 1 if self_retrieval else k

    assert results.shape[1] >= expected_result_size, \
        "Not enough retrieved results to evaluate Recall@{}".format(k)

    recall_at_k = np.zeros((k,))
    for i in range(len(query_labels)):
        pos = 0
        j = 0
        while pos < k:
            if self_retrieval and i == results[i, j]:
                j += 1
                continue
            if query_labels[i] == db_labels[results[i, j]]:
                recall_at_k[pos:] += 1
                break
            j += 1
            pos += 1

    return recall_at_k / float(len(query_labels)) * 100.0

def evaluate_float_binary_embedding_faiss(query_embeddings, db_embeddings, query_labels, db_labels,
                                          output, k=1000):
    """
        Wrapper function to evaluate Recall@k for float and binary embeddings
        Outputs recall@k strings for Cars, CUBS, Stanford Online Product, and InShop datasets
    """
    # Float embedding evaluation
    dists, retrieved_result_indices = _retrieve_knn_faiss_inner_product(query_embeddings, db_embeddings, k)
    r_at_k_f = evaluate_recall_at_k(dists, retrieved_result_indices, query_labels, db_labels, k)

    output_file = output + '_identity.eval'
    cars_cub_eval_str = "R@1, R@2, R@4, R@8: {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_f[0], r_at_k_f[1], r_at_k_f[3], r_at_k_f[7])
    sop_eval_str = "R@1, R@10, R@100, R@1000: {:.2f} & {:.2f} & {:.2f} & {:.2f}  \\\\".format(
        r_at_k_f[0], r_at_k_f[9], r_at_k_f[99], r_at_k_f[999])
    in_shop_eval_str = "R@1, R@10, R@20, R@30, R@40, R@50: {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_f[0], r_at_k_f[9], r_at_k_f[19], r_at_k_f[29], r_at_k_f[39], r_at_k_f[49])

    print(cars_cub_eval_str)
    print(sop_eval_str)
    print(in_shop_eval_str)
    with open(output_file, 'w') as of:
        of.write(cars_cub_eval_str + '\n')
        of.write(sop_eval_str + '\n')
        of.write(in_shop_eval_str + '\n')

    # Binary embedding evaluation
    binary_query_embeddings = np.require(query_embeddings > 0, dtype='float32')
    binary_db_embeddings = np.require(db_embeddings > 0, dtype='float32')

    dists, retrieved_result_indices = _retrieve_knn_faiss_euclidean(binary_query_embeddings, binary_db_embeddings, k)
    r_at_k_b = evaluate_recall_at_k(dists, retrieved_result_indices, query_labels, db_labels, k)

    output_file = output + '_binary.eval'
    cars_cub_eval_str = "R@1, R@2, R@4, R@8: {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_b[0], r_at_k_b[1], r_at_k_b[3], r_at_k_b[7])
    sop_eval_str = "R@1, R@10, R@100, R@1000: {:.2f} & {:.2f} & {:.2f} & {:.2f}  \\\\".format(
        r_at_k_b[0], r_at_k_b[9], r_at_k_b[99], r_at_k_b[999])
    in_shop_eval_str = "R@1, R@10, R@20, R@30, R@40, R@50: {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_b[0], r_at_k_b[9], r_at_k_b[19], r_at_k_b[29], r_at_k_b[39], r_at_k_b[49])

    print(cars_cub_eval_str)
    print(sop_eval_str)
    print(in_shop_eval_str)
    with open(output_file, 'w') as of:
        of.write(cars_cub_eval_str + '\n')
        of.write(sop_eval_str + '\n')
        of.write(in_shop_eval_str + '\n')

    return max(r_at_k_f[0], r_at_k_b[0])
