import torch
import numpy as np

def retrieve_with_budget(enc_A, index, mem_S, mem_l, X, config, budget):

    with torch.no_grad():

        print("\nGLOBAL RETRIEVAL DEBUG START")

        e, _ = enc_A(X)
        e = torch.nn.functional.normalize(e, dim=1)

        B = e.shape[0]

        base_k = config.get("retrieval_k_min", 50)
        max_k = 1000 # 500 - default setting
        step = 50

        print(f"[INFO] B={B}, budget={budget}")

        all_outputs = []
        all_locations = []
        all_distances = []

        global_used_ids = set()
        per_sample_unique_counts = []

        for i in range(B):

            print(f"\nSample {i}")

            local_unique = set()
            collected = []
            collected_loc = []
            collected_dist = []

            search_k = base_k

            while len(collected) < budget and search_k <= max_k:

                print(f"[SEARCH] sample={i}, search_k={search_k}")

                D, K = index.search(
                    e[i:i+1].cpu().numpy().astype(np.float32),
                    search_k
                )

                added_this_round = False

                for rank, idx in enumerate(K[0]):

                    # if idx in global_used_ids:
                    #     continue

                    if idx not in local_unique:
                        local_unique.add(idx)
                        global_used_ids.add(idx)

                        collected.append(mem_S[idx])
                        collected_loc.append(mem_l[idx])
                        collected_dist.append(D[0][rank])

                        added_this_round = True

                    if len(collected) >= budget:
                        break

                print(f"[INFO] collected={len(collected)}/{budget}")

                search_k += step

            # =====================================================
            # FALLBACK 
            # =====================================================
            if len(collected) == 0:
                fallback = K[0][0]
                collected.append(mem_S[fallback])
                collected_loc.append(mem_l[fallback])
                collected_dist.append(D[0][0])
                global_used_ids.add(fallback)

            # =====================================================
            # PAD (RANDOM from collected)
            # =====================================================
            n = len(collected)

            if n < budget:
                fill_n = budget - n

                idxs = np.random.choice(n, size=fill_n, replace=True)

                collected += [collected[i] for i in idxs]
                collected_loc += [collected_loc[i] for i in idxs]
                collected_dist += [collected_dist[i] for i in idxs]

            # =====================================================
            # CLIP TO REQUESTED
            # =====================================================
            collected = collected[:budget]
            collected_loc = collected_loc[:budget]
            collected_dist = collected_dist[:budget]

            per_sample_unique_counts.append(len(local_unique))

            all_outputs.append(np.stack(collected))
            all_locations.append(np.stack(collected_loc))
            all_distances.append(np.array(collected_dist))

            print(f"[FINAL SAMPLE SIZE] {len(collected)}")

        # =========================================================
        # OUTPUT
        # =========================================================
        y_ret = torch.tensor(np.stack(all_outputs)).float()
        loc_ret = torch.tensor(np.stack(all_locations)).float()
        dist_ret = torch.tensor(np.stack(all_distances)).float()

        requested_total = B * budget
        actual_unique = len(global_used_ids)

        print("\n DIVERSITY ANALYSIS")
        print(f"[REQUESTED UNIQUE SAMPLES] {requested_total}")
        print(f"[ACTUAL UNIQUE RETRIEVED]   {actual_unique}")
        print(f"[DUPLICATION RATE]          {1 - actual_unique/(requested_total+1e-8):.2%}")

        total_memory = len(mem_S)
        print("\n MEMORY COVERAGE ")
        print(f"[MEMORY BANK SIZE] {total_memory}")
        print(f"[COVERAGE] {(100*actual_unique/total_memory):.2f}%")

        print("\n PER-SAMPLE STATS ")
        print(f"[MEAN UNIQUE] {np.mean(per_sample_unique_counts):.2f} / {budget}")

        print("\n")

        return y_ret, loc_ret, dist_ret, {
            "target_original_count": len(X),
            "target_duplicated_count": requested_total - actual_unique
        }
    
def retrieve_with_coords(
    coordinates,
    season_start,
    season_end,
    crop_name,
    model_dir,
    budget=10
):
    checkpoint = torch.load(
        f"{model_dir}/{crop_name}.pth",
        map_location="cpu"
    )

    memory_bank = checkpoint["memory_bank"]
    coord_index = memory_bank["coord_index"]
    mem_S = memory_bank["mem_S"]
    mem_l = memory_bank["mem_l"]
    mem_season_start = memory_bank["mem_season_start"]
    mem_season_end = memory_bank["mem_season_end"]

    def season_contains(center, start, end):
        if start <= end:
            return start <= center <= end
        else:
            return center >= start or center <= end

    if season_start <= season_end:
        query_center = (season_start + season_end) / 2
    else:
        query_center = ((season_start + season_end + 365) / 2) % 365

    coords_rad = np.radians(
        np.asarray(coordinates)
    ).astype(np.float32)

    search_k = max(budget * 10, 100)

    distances, indices = coord_index.query(
        coords_rad,
        k=search_k
    )

    final_indices = []
    final_distances = []

    for i in range(indices.shape[0]):

        selected_idx = []
        selected_dist = []

        for d, idx in zip(distances[i], indices[i]):

            if season_contains(
                query_center,
                mem_season_start[idx],
                mem_season_end[idx]
            ):
                selected_idx.append(idx)
                selected_dist.append(d)

            if len(selected_idx) >= budget:
                break

        if len(selected_idx) == 0:
            selected_idx = indices[i][:budget].tolist()
            selected_dist = distances[i][:budget].tolist()

        final_indices.append(selected_idx[:budget])
        final_distances.append(selected_dist[:budget])

    final_indices = np.array(final_indices)
    final_distances = np.array(final_distances)

    return {
        "crop": crop_name,
        "spectra": mem_S[final_indices],
        "locations": mem_l[final_indices],
        "distance_km": final_distances * 6371,
        "indices": final_indices,
        "season_start": mem_season_start[final_indices],
        "season_end": mem_season_end[final_indices]
    }

def retrieve_with_aev(
    coordinates,
    X,
    crop_name,
    model_dir,
    budget=10
):
    checkpoint = torch.load(
        f"{model_dir}/{crop_name}.pth",
        map_location="cpu"
    )

    enc_in = checkpoint["enc_in"]
    memory_bank = checkpoint["memory_bank"]
    mem_S = memory_bank["mem_S"]
    mem_eA = memory_bank["mem_eA"]
    aev_index = memory_bank["aev_index"]
    mem_l = memory_bank["mem_l"]
    
    if not torch.is_tensor(X):
        X = torch.tensor(
            X,
            dtype=torch.float32
        )

    enc_in.eval()

    y_ret, loc_ret, dist_ret, stats = retrieve_with_budget(
        enc_in,
        aev_index,
        mem_S,
        mem_l,
        X,
        {},
        budget
    )

    return {
        "crop": crop_name,
        "spectra": y_ret.numpy(),
        "locations": loc_ret.numpy(),
        "similarity": dist_ret.numpy(),
        "stats": stats
    }

def retrieve_inference(
    raster_coords,
    raster_X,
    crop_name,
    model_dir,
    budget=10,
    config=None
):
    checkpoint = torch.load(
        f"{model_dir}/{crop_name}.pth",
        map_location="cpu"
    )

    enc_in = checkpoint["enc_in"]
    memory_bank = checkpoint["memory_bank"]

    aev_index = memory_bank["aev_index"]
    mem_S = memory_bank["mem_S"]
    mem_l = memory_bank["mem_l"]

    if config is None:
        config = {}

    X = torch.tensor(
        raster_X
    ).float()

    spectra, locations, similarity, stats = retrieve_with_budget(
        enc_in,
        aev_index,
        mem_S,
        mem_l,
        X,
        config,
        budget
    )

    spectra = spectra.numpy()
    locations = locations.numpy()
    similarity = similarity.numpy()

    # crop-conditioned expected spectral trajectory
    conditioned_spectra = np.mean(
        spectra,
        axis=1
    )

    return {
        "crop": crop_name,
        "coordinates": raster_coords,

        # (pixels, T, bands)
        "conditioned_spectra": conditioned_spectra,

        # retrieved neighbours
        "neighbor_spectra": spectra,
        "neighbor_locations": locations,

        # retrieval confidence
        "similarity": similarity,

        "stats": stats
    }