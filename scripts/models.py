import math
import os
import pandas as pd

# iglm
try:
    import iglm
    from iglm import IgLM
    iglm = IgLM()

except ImportError:
    pass

# antiberty
try:
    from antiberty import AntiBERTyRunner
    antiberty = AntiBERTyRunner()

except ImportError:
    pass

# progen
try:
    from progen_extra import *

except ImportError:
    pass

# esm2 (for sequence scoring)
try:
    import esm as esm_module
except ImportError:
    esm_module = None

# ablang2
try:
    import ablang2
except ImportError:
    ablang2 = None

# esm if
try:
    ## Verify that pytorch-geometric is correctly installed
    import torch_geometric
    import torch_sparse
    from torch_geometric.nn import MessagePassing

    # load model
    import esm
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # use eval mode for deterministic output e.g. without random dropout
    model = model.eval()

except ImportError:
    pass

# pyrsetta energy
try:
    import pyrosetta
    from pyrosetta.teaching import *
    pyrosetta.init()

except ImportError:
    pass

# proteinMPNN
try:
    from mpnn_extra import *

    # clone github repo
    sys.path.append('/home/mchungy1/scr4_jgray21/mchungy1/AbDesign/models/ProteinMPNN')
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

except ImportError:
    pass


def iglm_score(df):
    """
    input: df with columns: heavy,
                            light,
                            fitness
    output: df with columns: heavy,
                             light,
                             fitness,
                             heavy_perplexity,
                             light_perplexity,
                             average_perplexity
    """

    # score heavy sequences
    heavy_score = []

    for seq in df['heavy']:
        sequence = seq
        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"

        log_likelihood = iglm.log_likelihood(
            sequence,
            chain_token,
            species_token,
        )

        perplexity = math.exp(-log_likelihood)
        heavy_score.append(perplexity)

    df['heavy_perplexity'] = heavy_score

    # score light sequences
    light_score = []

    for seq in df['light']:
        sequence = seq
        chain_token = "[LIGHT]"
        species_token = "[HUMAN]"

        log_likelihood = iglm.log_likelihood(
            sequence,
            chain_token,
            species_token,
        )

        perplexity = math.exp(-log_likelihood)
        light_score.append(perplexity)

    df['light_perplexity'] = light_score

    df['average_perplexity'] = (df['heavy_perplexity'] + df['light_perplexity']) / 2

    return df


def antiberty_score(df, batch_size=16, device=None, enable_batch=True):
    # antibertyがインポートされていない場合は再試行
    global antiberty
    import torch

    # デバイスの決定
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    else:
        device = str(device)

    # デバイスオブジェクトに変換
    device_obj = torch.device(device)

    try:
        if 'antiberty' not in globals() or antiberty is None:
            from antiberty import AntiBERTyRunner
            antiberty = AntiBERTyRunner()

        # デバイスを設定
        # device属性はtorch.deviceオブジェクトとして設定
        if hasattr(antiberty, 'device'):
            antiberty.device = device_obj

        # モデルを指定デバイスに移動
        if hasattr(antiberty, 'model') and antiberty.model is not None:
            antiberty.model = antiberty.model.to(device_obj)

        # pseudo_log_likelihoodメソッドをモンキーパッチで修正
        # labelsテンソルをデバイスに移動するように修正
        original_pseudo_log_likelihood = antiberty.pseudo_log_likelihood

        def patched_pseudo_log_likelihood(sequences, batch_size=None):
            plls = []
            for s in sequences:
                masked_sequences = []
                for i in range(len(s)):
                    masked_sequence = list(s[:i]) + ["[MASK]"] + list(s[i + 1:])
                    masked_sequences.append(" ".join(masked_sequence))

                from antiberty.utils.general import exists
                tokenizer_out = antiberty.tokenizer(
                    masked_sequences,
                    return_tensors="pt",
                    padding=True,
                )
                tokens = tokenizer_out["input_ids"].to(antiberty.device)
                attention_mask = tokenizer_out["attention_mask"].to(antiberty.device)

                logits = []
                with torch.no_grad():
                    if not exists(batch_size):
                        batch_size_ = len(masked_sequences)
                    else:
                        batch_size_ = batch_size

                    from tqdm import tqdm
                    for i in tqdm(range(0, len(masked_sequences), batch_size_)):
                        batch_end = min(i + batch_size_, len(masked_sequences))
                        tokens_ = tokens[i:batch_end]
                        attention_mask_ = attention_mask[i:batch_end]

                        outputs = antiberty.model(
                            input_ids=tokens_,
                            attention_mask=attention_mask_,
                        )

                        logits.append(outputs.prediction_logits)

                logits = torch.cat(logits, dim=0)
                logits[:, :, antiberty.tokenizer.all_special_ids] = -float("inf")
                logits = logits[:, 1:-1]  # remove CLS and SEP tokens

                # get masked token logits
                logits = torch.diagonal(logits, dim1=0, dim2=1).unsqueeze(0)
                labels = antiberty.tokenizer.encode(
                    " ".join(list(s)),
                    return_tensors="pt",
                )[:, 1:-1].to(antiberty.device)  # デバイスに移動
                nll = torch.nn.functional.cross_entropy(
                    logits,
                    labels,
                    reduction="mean",
                )
                pll = -nll

                plls.append(pll)

            plls = torch.stack(plls, dim=0)

            return plls

        # モンキーパッチを適用
        antiberty.pseudo_log_likelihood = patched_pseudo_log_likelihood

    except (ImportError, NameError):
        raise ImportError(
            "AntiBERTyがインストールされていません。"
            "以下のコマンドでインストールしてください:\n"
            "pip install antiberty"
        )

    def compute_pll_batch_multi_seq(sequences_list, internal_batch_size=batch_size):
        """
        Process multiple sequences with batched masked position computation.
        sequences_list: list of sequences
        Returns: list of perplexities
        """
        all_plls = []

        # For each sequence, generate all masked variants
        all_masked_seqs = []
        seq_boundaries = [0]  # Track where each sequence's masked variants start
        original_seqs = []

        for s in sequences_list:
            masked_for_seq = []
            for i in range(len(s)):
                masked_sequence = list(s[:i]) + ["[MASK]"] + list(s[i + 1:])
                masked_for_seq.append(" ".join(masked_sequence))
            all_masked_seqs.extend(masked_for_seq)
            seq_boundaries.append(len(all_masked_seqs))
            original_seqs.append(s)

        if not all_masked_seqs:
            return [0.0] * len(sequences_list)

        # Tokenize all at once
        tokenizer_out = antiberty.tokenizer(
            all_masked_seqs,
            return_tensors="pt",
            padding=True,
        )
        tokens = tokenizer_out["input_ids"].to(device_obj)
        attention_mask = tokenizer_out["attention_mask"].to(device_obj)

        # Process in batches
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(all_masked_seqs), internal_batch_size):
                batch_end_idx = min(i + internal_batch_size, len(all_masked_seqs))
                tokens_ = tokens[i:batch_end_idx]
                attention_mask_ = attention_mask[i:batch_end_idx]

                outputs = antiberty.model(
                    input_ids=tokens_,
                    attention_mask=attention_mask_,
                )
                all_logits.append(outputs.prediction_logits)

        logits = torch.cat(all_logits, dim=0)
        logits[:, :, antiberty.tokenizer.all_special_ids] = -float("inf")
        logits = logits[:, 1:-1]  # remove CLS and SEP tokens

        # Calculate PLL for each sequence
        for seq_idx, s in enumerate(original_seqs):
            start = seq_boundaries[seq_idx]
            end = seq_boundaries[seq_idx + 1]

            if start == end:
                all_plls.append(0.0)
                continue

            seq_logits = logits[start:end]

            # Extract diagonal (masked position logits)
            # For each masked variant, get the logits at the masked position
            seq_logits_diag = torch.stack([seq_logits[i, i, :] for i in range(len(s))])

            labels = antiberty.tokenizer.encode(
                " ".join(list(s)),
                return_tensors="pt",
            )[:, 1:-1].to(device_obj)

            nll = torch.nn.functional.cross_entropy(
                seq_logits_diag.unsqueeze(0),
                labels,
                reduction="mean",
            )
            pll = -nll.item()
            all_plls.append(math.exp(-pll))

        return all_plls

    # Check if light chain exists
    has_light = 'light' in df.columns and df['light'].notna().any()

    heavy_score = []
    light_score = []

    if enable_batch:
        # Batch processing: process multiple sequences together
        from tqdm import tqdm
        for batch_start in tqdm(range(0, len(df), batch_size), desc="Scoring with AntiBERTy (batch)"):
            batch_end = min(batch_start + batch_size, len(df))

            # Collect all heavy sequences in batch
            batch_heavy_seqs = df['heavy'].iloc[batch_start:batch_end].tolist()
            heavy_ppls = compute_pll_batch_multi_seq(batch_heavy_seqs)
            heavy_score.extend(heavy_ppls)

            # Collect all light sequences in batch
            if has_light:
                batch_light_seqs = []
                batch_light_valid = []
                for i in range(batch_start, batch_end):
                    light_seq = df['light'].iloc[i]
                    if pd.notna(light_seq) and light_seq:
                        batch_light_seqs.append(light_seq)
                        batch_light_valid.append(True)
                    else:
                        batch_light_valid.append(False)

                if batch_light_seqs:
                    light_ppls = compute_pll_batch_multi_seq(batch_light_seqs)
                    ppl_idx = 0
                    for valid in batch_light_valid:
                        if valid:
                            light_score.append(light_ppls[ppl_idx])
                            ppl_idx += 1
                        else:
                            light_score.append(None)
                else:
                    light_score.extend([None] * (batch_end - batch_start))
            else:
                light_score.extend([None] * (batch_end - batch_start))
    else:
        # Sequential processing: process one row at a time
        for row in range(len(df)):
            sequences = [
                df['heavy'][row],
                df['light'][row] if has_light else "",
            ]

            pll = antiberty.pseudo_log_likelihood(sequences, batch_size=batch_size)

            perplexity_h = math.exp(-pll.tolist()[0])
            heavy_score.append(perplexity_h)

            if has_light and pd.notna(df['light'][row]) and df['light'][row]:
                perplexity_l = math.exp(-pll.tolist()[1])
                light_score.append(perplexity_l)
            else:
                light_score.append(None)

    df['heavy_perplexity'] = heavy_score

    if has_light:
        df['light_perplexity'] = light_score
        df['average_perplexity'] = df.apply(
            lambda row: (row['heavy_perplexity'] + row['light_perplexity']) / 2
            if row['light_perplexity'] is not None
            else row['heavy_perplexity'],
            axis=1
        )
    else:
        df['light_perplexity'] = None
        df['average_perplexity'] = df['heavy_perplexity']

    return df

def progen_score(df, model_version, device):
    ### main
    # Check if progen_extra is available
    try:
        from progen_extra import set_env, set_seed, create_model, create_tokenizer_custom, print_time, cross_entropy, log_likelihood
    except ImportError as e:
        raise ImportError(
            "progen_extra module could not be imported. "
            "Please ensure that progen dependencies are installed and the required paths are configured. "
            f"Original error: {e}"
        )

    # (0) constants
    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B

    # (2) preamble
    set_env()
    set_seed(42, deterministic=True)

    ### WILL HAVE TO EDIT TO MAKE GPU A FLAG
    if torch.cuda.is_available():
        print('gpu is available')

    else:
        print('falling back to cpu')

    device = torch.device(device)
    ckpt = f"/home/mchungy1/scr16_jgray21/mchungy1/progen/progen2/checkpoints/progen2-{model_version}"

    if device.type == 'cpu':
        print('falling back to fp32')
        fp16 = False

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=True).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='/home/mchungy1/scr16_jgray21/mchungy1/progen/progen2/tokenizer.json')

    def ce(tokens):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=fp16):
                target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                logits = model(target, labels=target).logits

                # shift
                logits = logits[:-1, ...]
                target = target[1:]

                return cross_entropy(logits=logits, target=target).item()

    def ll(tokens, f=log_likelihood, reduction='mean'):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                logits = model(target, labels=target).logits

                # shift
                logits = logits[:-1, ...]
                target = target[1:]

                # remove terminals
                bos_token, eos_token = 3, 4
                if target[-1] in [bos_token, eos_token]:
                    logits = logits[:-1, ...]
                    target = target[:-1]

                assert (target == bos_token).sum() == 0
                assert (target == eos_token).sum() == 0

                # remove unused logits
                first_token, last_token = 5, 29
                logits = logits[:, first_token:(last_token+1)]
                target = target - first_token

                assert logits.shape[1] == (last_token - first_token + 1)

                return f(logits=logits, target=target, reduction=reduction).item()

    # score heavy sequences using progen
    perplexity_mean_list_h = []

    for seq in df['heavy']:
        context = seq

        reverse = lambda s: s[::-1]

        ll_lr_mean = ll(tokens=context, reduction='mean')
        ll_rl_mean = ll(tokens=reverse(context), reduction='mean')

        ll_mean = .5 * (ll_lr_mean + ll_rl_mean)

        perplexity = math.exp(-ll_mean)

        perplexity_mean_list_h.append(perplexity)

    df['heavy_perplexity'] = perplexity_mean_list_h

    # score light sequences using progen
    perplexity_mean_list_l = []

    for seq in df['light']:
        context = seq

        reverse = lambda s: s[::-1]

        ll_lr_mean = ll(tokens=context, reduction='mean')
        ll_rl_mean = ll(tokens=reverse(context), reduction='mean')

        ll_mean = .5 * (ll_lr_mean + ll_rl_mean)

        perplexity = math.exp(-ll_mean)

        perplexity_mean_list_l.append(perplexity)

    df['light_perplexity'] = perplexity_mean_list_l

    df['average_perplexity'] = (df['heavy_perplexity'] + df['light_perplexity']) / 2

    return df

def esmif_score(pdb):

    # load pdb
    pdb_path = pdb
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

    # load chains
    fpath = pdb_path
    chain_ids = ['H', 'L']
    structure = esm.inverse_folding.util.load_structure(fpath, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    # conditional sequence log-likelihoods for given BB coordinates
    target_chain_id = 'H'
    target_seq = native_seqs[target_chain_id]
    ll_h, ll_withcoord = esm.inverse_folding.multichain_util.score_sequence_in_complex(
        model, alphabet, coords, target_chain_id, target_seq, padding_length=10)

    target_chain_id = 'L'
    target_seq = native_seqs[target_chain_id]
    ll_l, ll_withcoord = esm.inverse_folding.multichain_util.score_sequence_in_complex(
        model, alphabet, coords, target_chain_id, target_seq, padding_length=10)

    ll_avg =(ll_l + ll_h) / 2

    perplexity = math.exp(-ll_avg)

    return perplexity

def pyrosetta_score(pdb):

    # load pdb into pose
    pose = pyrosetta.pose_from_pdb(pdb)

    sfxn = get_score_function(True)

    return(sfxn(pose))

def mpnn_score(pdb):

    pdb_path = pdb

    # SETUP MODEL

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model_name = "v_48_020" #@param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

    # standard deviation of Gaussian noise to add to backbone atoms
    backbone_noise=0.00

    path_to_model_weights='/home/mchungy1/scr4_jgray21/mchungy1/AbDesign/models/ProteinMPNN/vanilla_model_weights'
    hidden_dim = 128
    num_layers = 3
    model_folder_path = path_to_model_weights
    if model_folder_path[-1] != '/':
        model_folder_path = model_folder_path + '/'
    checkpoint_path = model_folder_path + f'{model_name}.pt'

    checkpoint = torch.load(checkpoint_path, map_location=device)

    noise_level_print = checkpoint['noise_level']

    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    homomer = False #@param {type:"boolean"}
    designed_chain = "H L" #@param {type:"string"}
    fixed_chain = "" #@param {type:"string"}

    if designed_chain == "":
        designed_chain_list = []
    else:
        designed_chain_list = re.sub("[^A-Za-z]+",",", designed_chain).split(",")

    if fixed_chain == "":
        fixed_chain_list = []
    else:
        fixed_chain_list = re.sub("[^A-Za-z]+",",", fixed_chain).split(",")

    chain_list = list(set(designed_chain_list + fixed_chain_list))

    num_seqs = 1 #@param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
    num_seq_per_target = num_seqs

    #@markdown - Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly.
    sampling_temp = "0.1" #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"]

    save_score=0                      # 0 for False, 1 for True; save score=-log_prob to npy files
    save_probs=0                      # 0 for False, 1 for True; save MPNN predicted probabilites per position
    score_only=0                      # 0 for False, 1 for True; score input backbone-sequence pairs
    conditional_probs_only=0          # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
    conditional_probs_only_backbone=0 # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)

    batch_size=1                      # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
    max_length=20000                  # Max sequence length

    out_folder='.'                    # Path to a folder to output sequences, e.g. /home/out/
    jsonl_path=''                     # Path to a folder with parsed pdb into jsonl
    omit_AAs='X'                      # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.

    pssm_multi=0.0                    # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
    pssm_threshold=0.0                # A value between -inf + inf to restric per position AAs
    pssm_log_odds_flag=0               # 0 for False, 1 for True
    pssm_bias_flag=0                   # 0 for False, 1 for True

    folder_for_outputs = out_folder

    NUM_BATCHES = num_seq_per_target//batch_size
    BATCH_COPIES = batch_size
    temperatures = [float(item) for item in sampling_temp.split()]
    omit_AAs_list = omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

    chain_id_dict = None
    fixed_positions_dict = None
    pssm_dict = None
    omit_AA_dict = None
    bias_AA_dict = None
    tied_positions_dict = None
    bias_by_res_dict = None
    bias_AAs_np = np.zeros(len(alphabet))

    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)

    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

    for chain in chain_list:
        l = len(pdb_dict_list[0][f"seq_chain_{chain}"])

    if homomer:
        tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
    else:
        tied_positions_dict = None

    # RUN
    with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):
            score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict)
            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
            name_ = batch_clones[0]['name']

            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask*chain_M*chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss)
            native_score = scores.cpu().data.numpy()

            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    if tied_positions_dict == None:
                        sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), bias_by_res=bias_by_res_all)
                        S_sample = sample_dict["S"]
                    else:
                        sample_dict = model.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res_all)
                    # Compute scores
                        S_sample = sample_dict["S"]
                    log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
                    all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                    all_log_probs_list.append(log_probs.cpu().data.numpy())
                    S_sample_list.append(S_sample.cpu().data.numpy())
                    for b_ix in range(BATCH_COPIES):
                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
                        masked_list = masked_list_list[b_ix]
                        seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        score = scores[b_ix]
                        score_list.append(score)
                        native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                        if b_ix == 0 and j==0 and temp==temperatures[0]:
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(native_seq[start:end])
                                start = end
                            native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                            l0 = 0
                            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                l0 += mc_length
                                native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                l0 += 1
                            sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                            print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                            sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                            print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                            native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(seq[start:end])
                            start = end

                        seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                        l0 = 0
                        for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                            l0 += mc_length
                            seq = seq[:l0] + '/' + seq[l0:]
                            l0 += 1
                        score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                        seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)

    perplexity = math.exp(score)

    return perplexity


def esm2_score(df, model_name='esm2_t33_650M_UR50D', device=None, batch_size=4, ism_weights_path=None, enable_batch=True):
    """
    Score sequences using ESM2 or ISM models with pseudo-log-likelihood.

    Args:
        df: DataFrame with 'heavy' and optionally 'light' columns
        model_name: 'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', or 'ism'
        device: cuda/cpu device
        batch_size: batch size for processing sequences (when enable_batch=True) or masked positions (when enable_batch=False)
        ism_weights_path: path to ISM weights (only used when model_name='ism')
        enable_batch: if True, batch multiple sequences together; if False, process one-by-one

    Returns:
        df with heavy_perplexity, light_perplexity (if applicable), average_perplexity
    """
    import torch
    from tqdm import tqdm

    if esm_module is None:
        raise ImportError("ESM is not installed. Please install with: pip install fair-esm")

    # Device setup
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(device)

    # Load model
    if model_name == 'esm2_t33_650M_UR50D':
        esm2_model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
    elif model_name == 'esm2_t36_3B_UR50D':
        esm2_model, alphabet = esm_module.pretrained.esm2_t36_3B_UR50D()
    elif model_name == 'ism':
        # ISM uses ESM2 architecture with different weights
        esm2_model, alphabet = esm_module.pretrained.esm2_t33_650M_UR50D()
        if ism_weights_path is not None:
            ckpt = torch.load(ism_weights_path, map_location=device_obj)
            esm2_model.load_state_dict(ckpt)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    esm2_model = esm2_model.to(device_obj)
    esm2_model.eval()

    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    def compute_pseudo_log_likelihood(sequence, seq_batch_size=batch_size):
        """Compute pseudo-log-likelihood for a single sequence."""
        # Tokenize the original sequence
        data = [("seq", sequence)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device_obj)

        seq_len = len(sequence)
        log_probs_sum = 0.0

        # Process in batches of masked positions
        for start_pos in range(0, seq_len, seq_batch_size):
            end_pos = min(start_pos + seq_batch_size, seq_len)
            batch_positions = list(range(start_pos, end_pos))

            # Create masked versions for each position in this batch
            masked_tokens_list = []
            for pos in batch_positions:
                masked = tokens.clone()
                # Position in tokens is offset by 1 due to BOS token
                masked[0, pos + 1] = mask_idx
                masked_tokens_list.append(masked)

            masked_batch = torch.cat(masked_tokens_list, dim=0)

            with torch.no_grad():
                results = esm2_model(masked_batch, repr_layers=[], return_contacts=False)
                logits = results["logits"]

            # Extract log probabilities for the masked positions
            for i, pos in enumerate(batch_positions):
                # Get logits at the masked position (offset by 1 for BOS)
                pos_logits = logits[i, pos + 1, :]
                log_probs = torch.log_softmax(pos_logits, dim=-1)

                # Get the true token
                true_token = tokens[0, pos + 1]
                log_prob = log_probs[true_token].item()
                log_probs_sum += log_prob

        # Return average log probability
        return log_probs_sum / seq_len

    def compute_pseudo_log_likelihood_batch(sequences, masked_batch_size=batch_size):
        """Compute pseudo-log-likelihood for multiple sequences with batching."""
        all_plls = []

        # Tokenize all sequences together
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, all_tokens = batch_converter(data)
        all_tokens = all_tokens.to(device_obj)

        for seq_idx, sequence in enumerate(sequences):
            seq_len = len(sequence)
            log_probs_sum = 0.0

            # Create all masked variants for this sequence
            masked_tokens_list = []
            for pos in range(seq_len):
                masked = all_tokens[seq_idx:seq_idx+1].clone()
                # Position in tokens is offset by 1 due to BOS token
                masked[0, pos + 1] = mask_idx
                masked_tokens_list.append(masked)

            # Process masked variants in sub-batches
            for start in range(0, len(masked_tokens_list), masked_batch_size):
                end = min(start + masked_batch_size, len(masked_tokens_list))
                batch_masked = torch.cat(masked_tokens_list[start:end], dim=0)

                with torch.no_grad():
                    results = esm2_model(batch_masked, repr_layers=[], return_contacts=False)
                    logits = results["logits"]

                for i, pos in enumerate(range(start, end)):
                    pos_logits = logits[i, pos + 1, :]
                    log_probs = torch.log_softmax(pos_logits, dim=-1)
                    true_token = all_tokens[seq_idx, pos + 1]
                    log_probs_sum += log_probs[true_token].item()

            all_plls.append(log_probs_sum / seq_len)

        return all_plls

    # Check if light chain exists
    has_light = 'light' in df.columns and df['light'].notna().any()

    heavy_scores = []
    light_scores = []

    if enable_batch:
        # Batch processing: process multiple sequences together
        for batch_start in tqdm(range(0, len(df), batch_size), desc=f"Scoring with {model_name} (batch)"):
            batch_end = min(batch_start + batch_size, len(df))

            # Process heavy chains
            batch_heavy_seqs = df['heavy'].iloc[batch_start:batch_end].tolist()
            batch_heavy_plls = compute_pseudo_log_likelihood_batch(batch_heavy_seqs)
            heavy_scores.extend([math.exp(-pll) for pll in batch_heavy_plls])

            # Process light chains if exists
            if has_light:
                batch_light_seqs = []
                batch_light_valid = []
                for i in range(batch_start, batch_end):
                    light_seq = df['light'].iloc[i]
                    if pd.notna(light_seq) and light_seq:
                        batch_light_seqs.append(light_seq)
                        batch_light_valid.append(True)
                    else:
                        batch_light_valid.append(False)

                if batch_light_seqs:
                    batch_light_plls = compute_pseudo_log_likelihood_batch(batch_light_seqs)
                    pll_idx = 0
                    for valid in batch_light_valid:
                        if valid:
                            light_scores.append(math.exp(-batch_light_plls[pll_idx]))
                            pll_idx += 1
                        else:
                            light_scores.append(None)
                else:
                    light_scores.extend([None] * (batch_end - batch_start))
    else:
        # Sequential processing: process one sequence at a time
        for idx in tqdm(range(len(df)), desc=f"Scoring with {model_name}"):
            # Score heavy chain
            heavy_seq = df['heavy'].iloc[idx]
            heavy_pll = compute_pseudo_log_likelihood(heavy_seq)
            heavy_perplexity = math.exp(-heavy_pll)
            heavy_scores.append(heavy_perplexity)

            # Score light chain if exists
            if has_light:
                light_seq = df['light'].iloc[idx]
                if pd.notna(light_seq) and light_seq:
                    light_pll = compute_pseudo_log_likelihood(light_seq)
                    light_perplexity = math.exp(-light_pll)
                else:
                    light_perplexity = None
                light_scores.append(light_perplexity)

    df['heavy_perplexity'] = heavy_scores

    if has_light:
        df['light_perplexity'] = light_scores
        # Calculate average, handling None values
        df['average_perplexity'] = df.apply(
            lambda row: (row['heavy_perplexity'] + row['light_perplexity']) / 2
            if row['light_perplexity'] is not None
            else row['heavy_perplexity'],
            axis=1
        )
    else:
        df['light_perplexity'] = None
        df['average_perplexity'] = df['heavy_perplexity']

    return df


def ablang2_score(df, device=None, batch_size=16, enable_batch=True):
    """
    Score sequences using AbLang2 paired model with pseudo-log-likelihood.

    Args:
        df: DataFrame with 'heavy' and optionally 'light' columns
        device: cuda/cpu device
        batch_size: batch size for processing sequences (when enable_batch=True)
        enable_batch: if True, batch multiple sequences together; if False, process one-by-one

    Returns:
        df with heavy_perplexity, light_perplexity (if applicable), average_perplexity
    """
    import torch
    from tqdm import tqdm

    if ablang2 is None:
        raise ImportError("AbLang2 is not installed. Please install with: pip install ablang2")

    # Device setup
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Initialize AbLang2 paired model
    ablang_model = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device=device)

    # Get aa_to_token mapping
    aa_to_token = ablang_model.tokenizer.aa_to_token
    mask_char = '*'  # AbLang2 uses '*' as mask token

    # Check if light chain exists
    has_light = 'light' in df.columns and df['light'].notna().any()

    def compute_pseudo_log_likelihood_paired(heavy_seq, light_seq=None):
        """Compute pseudo-log-likelihood for paired or heavy-only sequence."""
        # Format sequence for AbLang2
        if light_seq is not None and pd.notna(light_seq):
            seq = f"{heavy_seq}|{light_seq}"
            heavy_len = len(heavy_seq)
            light_len = len(light_seq)
        else:
            seq = heavy_seq
            heavy_len = len(heavy_seq)
            light_len = 0

        # Total sequence length (including '|' separator if present)
        total_len = len(seq)

        # Compute pseudo-log-likelihood by masking each position
        log_probs_heavy = 0.0
        log_probs_light = 0.0

        # Create list of positions to mask (skip the '|' separator)
        positions_to_mask = []
        for i, char in enumerate(seq):
            if char != '|':
                positions_to_mask.append(i)

        # Process in batches
        for start_idx in range(0, len(positions_to_mask), batch_size):
            end_idx = min(start_idx + batch_size, len(positions_to_mask))
            batch_positions = positions_to_mask[start_idx:end_idx]

            masked_seqs = []
            true_aas = []
            for pos in batch_positions:
                # Create masked sequence
                seq_list = list(seq)
                true_aa = seq_list[pos]
                seq_list[pos] = mask_char
                masked_seqs.append(''.join(seq_list))
                true_aas.append(true_aa)

            # Tokenize masked sequences
            masked_tokenized = ablang_model.tokenizer(masked_seqs, pad=True, w_extra_tkns=False, device=device)

            # Get likelihoods
            with torch.no_grad():
                likelihoods = ablang_model.AbLang(masked_tokenized)

            # Extract log-probs for masked positions
            for i, (pos, true_aa) in enumerate(zip(batch_positions, true_aas)):
                if true_aa in aa_to_token:
                    true_idx = aa_to_token[true_aa]
                    # Get log probability at the masked position
                    log_prob = torch.log_softmax(likelihoods[i, pos, :], dim=-1)[true_idx].item()

                    # Determine if this is heavy or light chain
                    # Heavy chain is before '|', light chain is after
                    if light_seq is None:
                        log_probs_heavy += log_prob
                    elif pos < heavy_len:
                        log_probs_heavy += log_prob
                    else:
                        log_probs_light += log_prob

        # Calculate perplexities
        heavy_pll = log_probs_heavy / heavy_len if heavy_len > 0 else 0
        heavy_perplexity = math.exp(-heavy_pll)

        if light_len > 0:
            light_pll = log_probs_light / light_len
            light_perplexity = math.exp(-light_pll)
        else:
            light_perplexity = None

        return heavy_perplexity, light_perplexity

    def compute_pll_batch(heavy_light_pairs):
        """
        Compute pseudo-log-likelihood for multiple sequence pairs at once.
        Based on reference implementation.
        """
        # Format sequences
        seqs = []
        heavy_lens = []
        for h, l in heavy_light_pairs:
            if l is not None and pd.notna(l):
                seqs.append(f"{h}|{l}")
                heavy_lens.append(len(h))
            else:
                seqs.append(h)
                heavy_lens.append(len(h))

        # Tokenize all sequences
        labels = ablang_model.tokenizer(seqs, pad=True, w_extra_tkns=False, device=device)

        # Get indices for non-special tokens for all sequences
        special_tokens_tensor = torch.tensor(ablang_model.tokenizer.all_special_tokens, device=device)
        idxs = (~torch.isin(labels, special_tokens_tensor)).nonzero()

        # Group indices by sequence
        seq_indices = {}
        for idx_tensor in idxs:
            seq_idx = idx_tensor[0].item()
            if seq_idx not in seq_indices:
                seq_indices[seq_idx] = []
            seq_indices[seq_idx].append(idx_tensor[1].item())

        # Create masked versions for all positions
        all_masked_tokens = []
        all_target_indices = []
        all_target_labels = []

        for seq_idx, token_indices in seq_indices.items():
            masked_tokens = labels[seq_idx].repeat(len(token_indices), 1)
            for num, idx in enumerate(token_indices):
                masked_tokens[num, idx] = ablang_model.tokenizer.mask_token
            all_masked_tokens.append(masked_tokens)
            all_target_indices.append(token_indices)
            all_target_labels.append(labels[seq_idx, token_indices])

        # Concatenate all masked tokens
        all_masked_tokens = torch.cat(all_masked_tokens, dim=0)

        # Forward pass with sub-batching to avoid OOM
        all_logits = []
        sub_batch_size = batch_size * 50  # Larger sub-batch for efficiency
        for i in range(0, len(all_masked_tokens), sub_batch_size):
            batch_end = min(i + sub_batch_size, len(all_masked_tokens))
            with torch.no_grad():
                logits = ablang_model.AbLang(all_masked_tokens[i:batch_end])
            all_logits.append(logits)
        logits = torch.cat(all_logits, dim=0)
        logits[:, :, ablang_model.tokenizer.all_special_tokens] = -float("inf")

        # Calculate PLL for each sequence
        heavy_plls = []
        light_plls = []
        start_idx = 0

        for seq_idx, (h, l) in enumerate(heavy_light_pairs):
            token_indices = seq_indices.get(seq_idx, [])
            heavy_len = heavy_lens[seq_idx]

            if len(token_indices) == 0:
                heavy_plls.append(0)
                light_plls.append(None)
                continue

            seq_logits = logits[start_idx:start_idx + len(token_indices)]
            seq_logits_diag = torch.stack([seq_logits[num, idx] for num, idx in enumerate(token_indices)])
            seq_labels = all_target_labels[seq_idx]

            # Separate heavy and light scores based on position
            heavy_log_probs = 0.0
            light_log_probs = 0.0
            heavy_count = 0
            light_count = 0

            for i, tok_idx in enumerate(token_indices):
                log_prob = torch.log_softmax(seq_logits_diag[i], dim=-1)[seq_labels[i]].item()
                if tok_idx < heavy_len:
                    heavy_log_probs += log_prob
                    heavy_count += 1
                else:
                    light_log_probs += log_prob
                    light_count += 1

            heavy_pll = heavy_log_probs / heavy_count if heavy_count > 0 else 0
            heavy_plls.append(math.exp(-heavy_pll))

            if light_count > 0:
                light_pll = light_log_probs / light_count
                light_plls.append(math.exp(-light_pll))
            else:
                light_plls.append(None)

            start_idx += len(token_indices)

        return heavy_plls, light_plls

    heavy_scores = []
    light_scores = []

    if enable_batch:
        # Batch processing: process multiple sequences together
        for batch_start in tqdm(range(0, len(df), batch_size), desc="Scoring with AbLang2 (batch)"):
            batch_end = min(batch_start + batch_size, len(df))

            # Prepare pairs
            pairs = []
            for i in range(batch_start, batch_end):
                heavy_seq = df['heavy'].iloc[i]
                light_seq = df['light'].iloc[i] if has_light else None
                pairs.append((heavy_seq, light_seq))

            h_ppls, l_ppls = compute_pll_batch(pairs)
            heavy_scores.extend(h_ppls)
            light_scores.extend(l_ppls)
    else:
        # Sequential processing: process one sequence at a time
        for idx in tqdm(range(len(df)), desc="Scoring with AbLang2"):
            heavy_seq = df['heavy'].iloc[idx]

            if has_light:
                light_seq = df['light'].iloc[idx]
                heavy_ppl, light_ppl = compute_pseudo_log_likelihood_paired(heavy_seq, light_seq)
            else:
                heavy_ppl, light_ppl = compute_pseudo_log_likelihood_paired(heavy_seq, None)

            heavy_scores.append(heavy_ppl)
            light_scores.append(light_ppl)

    df['heavy_perplexity'] = heavy_scores

    if has_light:
        df['light_perplexity'] = light_scores
        df['average_perplexity'] = df.apply(
            lambda row: (row['heavy_perplexity'] + row['light_perplexity']) / 2
            if row['light_perplexity'] is not None
            else row['heavy_perplexity'],
            axis=1
        )
    else:
        df['light_perplexity'] = None
        df['average_perplexity'] = df['heavy_perplexity']

    return df
