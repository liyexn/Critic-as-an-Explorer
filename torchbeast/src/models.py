# Original code Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the original license in the LICENSE file.
#
# Modified in 2025
#   - Added CAE, CAE+, CAE+ w/o U algorithms
#   - Simplified unrelated parts
#
# Additional modifications are released under CC BY-NC 4.0


import torch
from torch import nn 
from torch.nn import functional as F


from nle import nethack

NUM_CHARS = 256
PAD_CHAR = 0

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MiniHackInverseDynamicsAttentionNet(nn.Module):
    def __init__(self, num_actions, emb_size=128, p_dropout=0.0, weighted_emb_size=64):
        super(MiniHackInverseDynamicsAttentionNet, self).__init__()
        self.num_actions = num_actions

        init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        self.weighted_emb = init_relu_(nn.Linear(emb_size, weighted_emb_size))

        self.inverse_dynamics = nn.Sequential(
            init_relu_(nn.Linear(2 * weighted_emb_size, 256)),
            nn.ReLU(),
            nn.Dropout(p=p_dropout)
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(256, self.num_actions))

    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((self.weighted_emb(state_embedding), self.weighted_emb(next_state_embedding)), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits


class MiniHackInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions, emb_size=128, p_dropout=0.0):
        super(MiniHackInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions 
        
        init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        self.inverse_dynamics = nn.Sequential(
            init_relu_(nn.Linear(2 * emb_size, 256)),
            nn.ReLU(),
            nn.Dropout(p=p_dropout)
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(256, self.num_actions))

    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits


def _step_to_range(delta, num_steps):
    """ Range of `num_steps` integers with distance `delta` centered around zero """
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


class Crop(nn.Module):
    """ Helper class for NetHackNet below """

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[None, :].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[:, None].expand(-1, self.width_target)

        # "clone" necessary: https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x, y coordinates
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x, y coordinates
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)
        
        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class NetHackPolicyNet(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_actions,
        use_lstm,
        sphere_norm=0,
        hidden_dim=1024,
        embedding_dim=64,
        crop_dim=9,
        num_layers=5,
        msg_model="lt_cnn",
        weighted_emb_size=64
    ):
        super(NetHackPolicyNet, self).__init__()

        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))        

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]

        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.sphere_norm = sphere_norm

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = hidden_dim

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # CNN crop model
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model
        out_dim += self.crop_dim ** 2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.msg_model = msg_model
        if self.msg_model == 'lt_cnn':
            self.msg_hdim = 64
            self.msg_edim = 32
            self.char_lt = nn.Embedding(
                NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR
            )            
            self.conv1 = nn.Conv1d(
                self.msg_edim, self.msg_hdim, kernel_size=7
            )
            # remaining convolutions, relus, pools, and a small FC network
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv3
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv4
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv5
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv6
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                nn.ReLU(),
                nn.Linear(2 * self.msg_hdim, self.msg_hdim),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += self.msg_hdim
            
        elif self.msg_model == 'lt_cnn_small':
            self.msg_hdim = 64
            self.msg_edim = 32
            self.char_lt = nn.Embedding(
                NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR
            )            
            self.conv1 = nn.Conv1d(
                self.msg_edim, self.msg_hdim, kernel_size=7
            )
            # remaining convolutions, relus, pools, and a small FC network
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv3
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                nn.ReLU(),
                nn.Linear(2 * self.msg_hdim, self.msg_hdim),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += self.msg_hdim


        self.fc1 = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )

        self.weight_matrix_state = nn.Linear(self.h_dim, self.h_dim, bias=False)

        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)

        self.policy = nn.Linear(self.h_dim, self.num_actions)
        self.baseline = nn.Linear(self.h_dim, 1)


    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """ Maintains a running mean of reward """
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """ Returns standard deviation of the running mean of the reward """
        return torch.sqrt(self.reward_m2 / self.reward_count)        

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def _select(self, embed, x):
        # work around slow backward pass of nn.Embedding, see https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs, core_state, decode=False):

#        env_outputs = env_outputs['frame']
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [T x B x F]
        blstats = env_outputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        # FIXME: hack to use compatible blstats to before
        # blstats = blstats[:, [0, 1, 21, 10, 11]]

        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)
        
        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)


        # MESSAGING MODEL
        if self.msg_model != "none":
            # [T x B x 256] -> [T * B x 256]
            messages = env_outputs["message"].long().view(T * B, -1)
            if self.msg_model == "lt_cnn":
                # [ T * B x E x 256 ]
                char_emb = self.char_lt(messages).transpose(1, 2)
                char_rep = self.conv2_6_fc(self.conv1(char_emb))

            reps.append(char_rep)

        

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st1 = self.fc1(st)
        st = self.fc2(st1)
        if self.sphere_norm == 1:
            st = F.normalize(st, p=2, dim=-1)


        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~env_outputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                try:
                    output, core_state = self.core(input.unsqueeze(0), core_state)
                except:
                    print('self.core')
                    print(core_input)
                    print(self.core)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # do Not sample when testing
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        outputs = dict(policy_logits=policy_logits, baseline=baseline, action=action)
        outputs['policy_hiddens'] = st.detach()
        outputs['emb'] = st.view(T, B, -1)
        return (
            outputs,
            core_state,
        )




class NetHackStateEmbeddingNet(nn.Module):
    def __init__(
            self,
            observation_shape,
            use_lstm,
            hidden_dim=1024,
            embedding_dim=64,
            crop_dim=9,
            num_layers=5,
            msg_model="lt_cnn",
            p_dropout=0.0,
            weighted_emb_size=64
    ):
        super(NetHackStateEmbeddingNet, self).__init__()

        self.register_buffer("reward_sum", torch.zeros(()))
        self.register_buffer("reward_m2", torch.zeros(()))
        self.register_buffer("reward_count", torch.zeros(()).fill_(1e-8))

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]
        self.p_dropout = p_dropout

        self.use_lstm = use_lstm

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = hidden_dim

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.Sequential(nn.ELU(), nn.Dropout(self.p_dropout))] * len(conv_extract))
        )

        # CNN crop model
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.Sequential(nn.ELU(), nn.Dropout(self.p_dropout))] * len(conv_extract))
        )

        self.feat_extract = self.extract_representation

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim ** 2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
        )

        self.msg_model = msg_model
        if msg_model == 'lt_cnn':
            self.msg_hdim = 64
            self.msg_edim = 32
            self.char_lt = nn.Embedding(
                NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR
            )
            self.conv1 = nn.Conv1d(
                self.msg_edim, self.msg_hdim, kernel_size=7
            )
            # remaining convolutions, relus, pools, and a small FC network
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.Dropout2d(p=self.p_dropout),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv3
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.Dropout2d(p=self.p_dropout),
                # conv4
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.Dropout2d(p=self.p_dropout),
                # conv5
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.Dropout2d(p=self.p_dropout),
                # conv6
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.Dropout2d(p=self.p_dropout),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                nn.ReLU(),
                nn.Dropout(p=self.p_dropout),
                nn.Linear(2 * self.msg_hdim, self.msg_hdim),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += self.msg_hdim

        self.fc1 = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        self.weighted_emb = nn.Linear(self.h_dim, weighted_emb_size, bias=False)

        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)

    @torch.no_grad()
    def update_running_moments(self, reward_batch):
        """Maintains a running mean of reward."""
        new_count = len(reward_batch)
        new_sum = torch.sum(reward_batch)
        new_mean = new_sum / new_count

        curr_mean = self.reward_sum / self.reward_count
        new_m2 = torch.sum((reward_batch - new_mean) ** 2) + (
                (self.reward_count * new_count)
                / (self.reward_count + new_count)
                * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    @torch.no_grad()
    def get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        return torch.sqrt(self.reward_m2 / self.reward_count)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def _select(self, embed, x):
        # work around slow backward pass of nn.Embedding, see https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs, core_state):
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [T x B x F]
        blstats = env_outputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        # FIXME: hack to use compatible blstats to before
        # blstats = blstats[:, [0, 1, 21, 10, 11]]

        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        # MESSAGING MODEL
        if self.msg_model != "none":
            # [T x B x 256] -> [T * B x 256]
            messages = env_outputs["message"].long()
            messages = messages.view(T * B, -1)
            if self.msg_model == "lt_cnn":
                # [ T * B x E x 256 ]
                char_emb = self.char_lt(messages).transpose(1, 2)
                char_rep = self.conv2_6_fc(self.conv1(char_emb))

            reps.append(char_rep)

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st1 = self.fc1(st)
        st = self.fc2(st1)

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~env_outputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # reset core state to zero whenever an episode ended
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                try:
                    output, core_state = self.core(input.unsqueeze(0), core_state)
                except:
                    print('self.core')
                    print(core_input)
                    print(self.core)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        core_output = core_output.view(T, B, -1)

        return (
            core_output,
            core_state,
        )

    @torch.no_grad()
    def weight_forward(self, env_outputs, core_state):
        core_output, core_state = self.forward(env_outputs, core_state)
        core_output_weight = self.weighted_emb(core_output)
        return (
            core_output_weight,
            core_state,
        )



    

