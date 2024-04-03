import torch


class TrainPadSequence:
    def __call__(self, sorted_batch):
        sequences = [x[0] for x in sorted_batch]
        aud_sequences = [x[1] for x in sorted_batch]
        spec_dim = []

        for aud in aud_sequences:
            spec_dim.append(aud.shape[3])

        max_spec_dim = max(spec_dim)
        audio_features = torch.zeros(len(spec_dim), 16, 1, 64, max_spec_dim)
        for batch_idx, spectrogram in enumerate(aud_sequences):
            if spectrogram.shape[2] < max_spec_dim:
                audio_features[batch_idx, :, :, :, -spectrogram.shape[3] :] = (
                    spectrogram
                )
            else:
                audio_features[batch_idx, :, :, :, :] = spectrogram

        labelV = [x[2] for x in sorted_batch]
        labelA = [x[3] for x in sorted_batch]
        wavFile = [x[4] for x in sorted_batch]

        visual_sequences = torch.stack(sequences)
        labelsV = torch.stack(labelV)
        labelsA = torch.stack(labelA)

        return visual_sequences, audio_features, labelsV, labelsA, wavFile


class ValPadSequence:
    def __call__(self, sorted_batch):

        sequences = [x[0] for x in sorted_batch]
        aud_sequences = [x[1] for x in sorted_batch]
        spec_dim = []
        for aud in aud_sequences:
            spec_dim.append(aud.shape[3])

        max_spec_dim = max(spec_dim)
        audio_features = torch.zeros(len(spec_dim), 16, 1, 64, max_spec_dim)
        for batch_idx, spectrogram in enumerate(aud_sequences):
            if spectrogram.shape[2] < max_spec_dim:
                audio_features[batch_idx, :, :, :, -spectrogram.shape[3] :] = (
                    spectrogram
                )
            else:
                audio_features[batch_idx, :, :, :, :] = spectrogram

        frameids = [x[2] for x in sorted_batch]
        v_ids = [x[3] for x in sorted_batch]
        v_lengths = [x[4] for x in sorted_batch]
        labelV = [x[5] for x in sorted_batch]
        labelA = [x[6] for x in sorted_batch]
        wavFile = [x[7] for x in sorted_batch]

        visual_sequences = torch.stack(sequences)
        labelsV = torch.stack(labelV)
        labelsA = torch.stack(labelA)
        return (
            visual_sequences,
            audio_features,
            frameids,
            v_ids,
            v_lengths,
            labelsV,
            labelsA,
            wavFile,
        )


class TestPadSequence:
    def __call__(self, sorted_batch):

        sequences = [x[0] for x in sorted_batch]
        aud_sequences = [x[1] for x in sorted_batch]
        spec_dim = []
        for aud in aud_sequences:
            spec_dim.append(aud.shape[3])

        max_spec_dim = max(spec_dim)
        audio_features = torch.zeros(len(spec_dim), 16, 1, 64, max_spec_dim)
        for batch_idx, spectrogram in enumerate(aud_sequences):
            if spectrogram.shape[2] < max_spec_dim:
                audio_features[batch_idx, :, :, :, -spectrogram.shape[3] :] = (
                    spectrogram
                )
            else:
                audio_features[batch_idx, :, :, :, :] = spectrogram

        frameids = [x[2] for x in sorted_batch]
        v_ids = [x[3] for x in sorted_batch]
        v_lengths = [x[4] for x in sorted_batch]
        wavFile = [x[5] for x in sorted_batch]

        visual_sequences = torch.stack(sequences)

        return visual_sequences, audio_features, frameids, v_ids, v_lengths, wavFile
