<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/supported.html (원본) -->

# Supported Frameworks
We support almost all the commonly used Machine Learning frameworks & libraries such as ONNX, PyTorch, Keras, TensorFlow, and TensorFlow Lite.

![Supported deep-learning frameworks](res/image/supported_frameworks_new.png "Supported deep-learning frameworks")

# Mobilint IR Operations List
- The following list includes only the operations natively supported by Mobilint hardware. Operations not explicitly listed may still be partially supported during compilation via graph-level transformations into equivalent operations.
- All operations supported by Mobilint hardware support the `NHWC`(channel-last)-data format with a batch size of `1`.



| Operation | Notes |
| :--- | :--- |
| Abs |num inputs: 1<br>num outputs: 1<br> |
| AddPartial |num inputs: 2<br>num outputs: 1<br>attributes:<br>- is_input0_larger: bool |
| AddingConstant |num inputs: 1<br>num outputs: 1<br>attributes:<br>- constant: Weight<br>- is_const_first: bool (default=false)<br>- shape: int32[] (default=[]) |
| Adding |num inputs: 2<br>num outputs: 1<br> |
| Batchnorm |num inputs: 1<br>num outputs: 1<br>attributes:<br>- gamma: Weight<br>- beta: Weight<br>- moving_mean: Weight<br>- moving_var: Weight<br>- epsilon: Weight<br>- channel_last: bool (default=true) |
| Cast |num inputs: 1<br>num outputs: 1<br>attributes:<br>- dtype: str |
| Ceil |num inputs: 1<br>num outputs: 1<br> |
| Celu |num inputs: 1<br>num outputs: 1<br>attributes:<br>- alpha: float32 |
| Clip |num inputs: 1<br>num outputs: 1<br>attributes:<br>- clip_max_value: float32 (optional)<br>- clip_min_value: float32 (optional) |
| Concatenate |num inputs: variable<br>num outputs: 1<br>attributes:<br>- original_key: str (optional)<br>- axis: int32 (optional)<br>- original_axis: int32 (optional)<br>- original_inputs: int32[] (optional)<br>- input_index: int32[] (optional)<br>- input_names: str[] (optional)<br><br>description:<br>input_names: input activation names in original order. input_index: input index in computation order. |
| Convolution |num inputs: 1<br>num outputs: 1<br>attributes:<br>- in_channels: int32<br>- out_channels: int32<br>- h_kernel: int32<br>- w_kernel: int32<br>- h_stride: int32<br>- w_stride: int32<br>- h_dilation: int32<br>- w_dilation: int32<br>- west_padding: int32<br>- east_padding: int32<br>- north_padding: int32<br>- south_padding: int32<br>- padding_list: int32[] (default=[])<br>- group_number: int32<br>- weight: Weight<br>- bias: Weight (optional)<br>- channel_last: bool (default=true) |
| Convolution3d |num inputs: 1<br>num outputs: 1<br>attributes:<br>- in_channels: int32<br>- out_channels: int32<br>- h_kernel: int32<br>- w_kernel: int32<br>- d_kernel: int32<br>- h_stride: int32<br>- w_stride: int32<br>- d_stride: int32<br>- h_dilation: int32<br>- w_dilation: int32<br>- d_dilation: int32<br>- west_padding: int32<br>- east_padding: int32<br>- north_padding: int32<br>- south_padding: int32<br>- front_padding: int32<br>- back_padding: int32<br>- padding_list: int32[] (default=[])<br>- group_number: int32<br>- weight: Weight<br>- bias: Weight (optional)<br>- channel_last: bool (default=true) |
| Crop |num inputs: 1<br>num outputs: 1<br>attributes:<br>- w_offset: int32 (default=0)<br>- h_offset: int32 (default=0)<br>- crop_width: int32<br>- crop_height: int32 |
| DepthwiseConvolution |num inputs: 1<br>num outputs: 1<br>attributes:<br>- in_channels: int32<br>- out_channels: int32<br>- h_kernel: int32<br>- w_kernel: int32<br>- h_stride: int32<br>- w_stride: int32<br>- h_dilation: int32<br>- w_dilation: int32<br>- west_padding: int32<br>- east_padding: int32<br>- north_padding: int32<br>- south_padding: int32<br>- padding_list: int32[] (default=[])<br>- group_number: int32<br>- weight: Weight<br>- bias: Weight (optional)<br>- channel_last: bool (default=true) |
| DeviceBridge |num inputs: 1<br>num outputs: 1<br>attributes:<br>- squeeze_axes: int32[]<br>- unsqueeze_axes: int32[]<br>- permute_axes: int64[]<br>- device: str (optional) |
| DivConstant |num inputs: 1<br>num outputs: 1<br>attributes:<br>- constant: Weight<br>- is_const_first: bool<br>- shape: int32[] (default=[]) |
| Div |num inputs: 2<br>num outputs: 1<br> |
| Elu |num inputs: 1<br>num outputs: 1<br>attributes:<br>- leaky_alpha: float32 |
| Erf |num inputs: 1<br>num outputs: 1<br> |
| Exp |num inputs: 1<br>num outputs: 1<br> |
| Expand |num inputs: 1<br>num outputs: 1<br>attributes:<br>- constant: Weight (optional)<br>- is_const_first: bool |
| Gelu |num inputs: 1<br>num outputs: 1<br>attributes:<br>- approximate: bool (default=false) |
| GLU |num inputs: 1<br>num outputs: 1<br> |
| GridSample |num inputs: 2<br>num outputs: 1<br>attributes:<br>- align_corners: int32 (default=0)<br>- mode: str (default="linear")<br>- padding_mode: str (default="zeros")<br><br>description:<br>inputs: X, grid<br>X is in channel-last format.<br>X: (n, h, w, c)<br>grid: (n, h_out, w_out, num_spatial_dims). in this case, num_spatial_dims=2. |
| GroupConvolution |num inputs: 1<br>num outputs: 1<br>attributes:<br>- in_channels: int32<br>- out_channels: int32<br>- h_kernel: int32<br>- w_kernel: int32<br>- h_stride: int32<br>- w_stride: int32<br>- h_dilation: int32<br>- w_dilation: int32<br>- west_padding: int32<br>- east_padding: int32<br>- north_padding: int32<br>- south_padding: int32<br>- padding_list: int32[] (default=[])<br>- group_number: int32<br>- weight: Weight<br>- bias: Weight (optional)<br>- channel_last: bool (default=true) |
| GroupNormalization |num inputs: 1<br>num outputs: 1<br>attributes:<br>- num_groups: int32<br>- scale: Weight<br>- bias: Weight (optional)<br>- epsilon: Weight<br>- channel_last: bool (default=true) |
| HardSigmoid |num inputs: 1<br>num outputs: 1<br>attributes:<br>- alpha: float32 (optional)<br>- beta: float32 (optional) |
| Hardtanh |num inputs: 1<br>num outputs: 1<br>attributes:<br>- min_val: float32 (default=-1.0)<br>- max_val: float32 (default=1.0) |
| HardSwish |num inputs: 1<br>num outputs: 1<br> |
| Hardshrink |num inputs: 1<br>num outputs: 1<br>attributes:<br>- lambd: float32 (default=0.5) |
| Softshrink |num inputs: 1<br>num outputs: 1<br>attributes:<br>- lambd: float32 (default=0.5) |
| HeaderView |num inputs: 1<br>num outputs: 1<br>attributes:<br>- view_shape: int32[] (default=[])<br>- num_heads: int32 (default=-1)<br>- batch_order: int32 (default=0)<br><br>description:<br>HeaderView simplifies the following sequence of operations.<br>If batch_order == 0<br>(n,h,w,c)-[reshape]->(1, h, w, num_heads c/num_heads)<br>-[transpose01324]->(1, h, num_heads, w, c/num_heads)<br>-[reshape]->(1, h*num_heads, w, c/num_heads)<br>If batch_order == 1<br>(n,h,w,c)-[reshape]->(1, h, w, num_heads, c/num_heads)<br>-[transpose03124]->(1, num_heads, h, w, c/num_heads)<br>-[reshape]->(1, num_heads*h, w, c/num_heads) |
| HeaderViewRevert |num inputs: 1<br>num outputs: 1<br>attributes:<br>- revert_shape: int32[] (default=[])<br>- num_heads: int32 (default=-1)<br>- batch_order: int32 (default=0)<br><br>description:<br>HeaderViewRevert simplifies the following sequence of operations.<br>If batch_order == 0<br>(n,h,w,c)-[reshape]->(1, h/num_heads, num_heads, w, c)<br>-[transpose01324]->(1, h/num_heads, w, num_heads, c)<br>-[reshape]->(1, h/num_heads, w, num_heads * c)<br>If batch_order == 1<br>(n,h,w,c)-[reshape]->(1, num_heads, h/num_heads, w, c)<br>-[transpose02314]->(1, h/num_heads, w, num_heads, c)<br>-[reshape]->(1, h/num_heads, w, num_heads*c) |
| Identity |num inputs: 1<br>num outputs: 1<br> |
| InputConstant |num inputs: 0<br>num outputs: 1<br>attributes:<br>- constant: Weight<br>- dtype: str (default="")<br>- shape: int32[] (default=[]) |
| Input |num inputs: 1<br>num outputs: 1<br>attributes:<br>- orig_name: str (default="") |
| InstanceNormalization |num inputs: 1<br>num outputs: 1<br>attributes:<br>- scale: Weight<br>- bias: Weight (optional)<br>- epsilon: Weight<br>- channel_last: bool (default=true)<br>- num_groups: int32 (default=-1)<br>- group_order: int32 (default=-1)<br><br>description:<br>If `num_groups` and `order` are specified, the channel dimension is grouped and rearranged before normalization.<br>For example, the following sequence:<br>(n, 1, w, c*2)-[reshape]->(n, w, c, 2)<br>-[transpose0312]->(n, 2, w, c)<br>-[InstanceNorm, num_features=c]->(n, 2, w, c)<br>-[transpose0231]->(n, w, c, 2)<br>-[reshape]->(n, 1, w, c*2)<br>is equivalent to a single operation:<br>(n, 1, w, c*2)-[InstanceNormalization, num_features=c, num_groups=2, group_order=1]->(n, 1, w, c*2)<br>The follosing sequence:<br>(n, 1, w, 2*c)-[reshape]->(n,w,2,c)<br>-[InstanceNorm, num_features=c]->(n, w, 2, c)<br>-[reshape]v(n, 1, w, c*2)<br>is equivalent to a single operation:<br>(n, 1, w, 2*c)-[InstanceNormalization, num_features=c, num_groups=2, group_order=0]->(n, 1, w, 2*c) |
| L1Normalization |num inputs: 1<br>num outputs: 1<br>attributes:<br>- norm_axis: int32[] |
| L2Normalization |num inputs: 1<br>num outputs: 1<br>attributes:<br>- norm_axis: int32[]<br>- epsilon: Weight (optional) |
| SumNormalization |num inputs: 1<br>num outputs: 1<br>attributes:<br>- norm_axis: int32[]<br>- epsilon: float32 (optional) |
| LSTM |num inputs: variable<br>num outputs: 1<br>attributes:<br>- W: Weight<br>- R: Weight<br>- B: Weight (optional)<br>- lstm_group: str<br>- input_mask: str<br>- output_mask: str<br>- P: Weight (optional)<br>- direction: str (default="forward")<br>- batch_first: bool (default=false)<br>- hidden_size: int32<br><br>description:<br>inputs:[input, sequence_lens, initial_h, initial_c]<br>outputs:[output] or [hidden_out] or [cell_out]<br>input_mask: "ishc" or "i", etc,...<br>output_mask: "o" or "h" or "c" |
| LayerNormalization |num inputs: 1<br>num outputs: 1<br>attributes:<br>- normalized_shape: int64[] (default=[])<br>- scale: Weight<br>- bias: Weight (optional)<br>- epsilon: Weight |
| Logit |num inputs: 1<br>num outputs: 1<br>attributes:<br>- eps: float32 (optional)<br><br>description:<br>Compute y = ln(z / (1 - z)). If eps is given, we clamp input z = clamp(z, min=eps, max=1-eps) |
| RNN |num inputs: variable<br>num outputs: 1<br>attributes:<br>- W: Weight<br>- R: Weight<br>- B: Weight (optional)<br>- rnn_group: str<br>- input_mask: str<br>- output_mask: str<br>- direction: str (default="forward")<br>- batch_first: bool (default=false)<br>- hidden_size: int32<br><br>description:<br>inputs:[input, sequence_lens, initial_h]<br>outputs:[output] or [hidden_out]<br>input_mask: "ish" or "i", etc,...<br>output_mask: "o" or "h" |
| GRU |num inputs: variable<br>num outputs: 1<br>attributes:<br>- W: Weight<br>- R: Weight<br>- B: Weight (optional)<br>- gru_group: str<br>- input_mask: str<br>- output_mask: str<br>- direction: str (default="forward")<br>- batch_first: bool (default=false)<br>- linear_before_reset: bool (default=false)<br>- hidden_size: int32<br><br>description:<br>inputs:[input, sequence_lens, initial_h]<br>outputs:[output] or [hidden_out]<br>input_mask: "ish" or "i", etc,...<br>output_mask: "o" or "h" |
| LeakyRelu |num inputs: 1<br>num outputs: 1<br>attributes:<br>- leaky_alpha: float32 |
| Log |num inputs: 1<br>num outputs: 1<br> |
| LogSigmoid |num inputs: 1<br>num outputs: 1<br> |
| MaskSoftmax |num inputs: 1<br>num outputs: 1<br>attributes:<br>- beta: float32<br>- axis: int32 |
| MatMul |num inputs: variable<br>num outputs: 1<br>attributes:<br>- constant: Weight (optional)<br>- is_const_first: bool (default=false)<br>- repeat: int32[2] (default=[-1, -1])<br><br>description:<br>If repeat=(0, 4), then input0 is repeated 4 times along the h-direction.<br>For example, you can perform a matrix multiplication with input0 of shape (1, 8, 52, 64) and<br>input1 of shape (1, 32, 52, 64) by concatenating input0 4 times, resulting in a input0 shape of (1, 32, 52, 64). |
| Mish |num inputs: 1<br>num outputs: 1<br> |
| MultiplyConstant |num inputs: 1<br>num outputs: 1<br>attributes:<br>- constant: Weight<br>- is_const_first: bool (default=false)<br>- shape: int32[] (default=[]) |
| Multiply |num inputs: 2<br>num outputs: 1<br> |
| NonMaxSuppression |num inputs: 2<br>num outputs: variable<br>attributes:<br>- max_output_size: int64<br>- iou_threshold: float32<br>- score_threshold: float32<br>- soft_nms_sigma: float32<br>- pad_to_max_output_size: bool<br>- center_point_box: int32 (default=0) |
| Output |num inputs: 1<br>num outputs: 1<br>attributes:<br>- orig_name: str (default="") |
| PRelu |num inputs: 1<br>num outputs: 1<br>attributes:<br>- leaky_alpha: Weight |
| Pad |num inputs: 1<br>num outputs: 1<br>attributes:<br>- west_padding: int32 (default=-1)<br>- east_padding: int32 (default=-1)<br>- south_padding: int32 (default=-1)<br>- north_padding: int32 (default=-1)<br>- pad_mode: str<br>- constant: float32 (default=0.0)<br>- padding_list: int32[] (optional)<br>- channel_last: bool (default=true)<br><br>description:<br>padding_list format should be [x1_begin, x2_begin, …, x1_end, x2_end,…],<br>where xi_begin is the number of pad values added at the beginning of axis |
| Pooling |num inputs: 1<br>num outputs: 1<br>attributes:<br>- h_kernel: int32<br>- w_kernel: int32<br>- h_stride: int32<br>- w_stride: int32<br>- h_dilation: int32 (default=1)<br>- w_dilation: int32 (default=1)<br>- west_padding: int32<br>- east_padding: int32<br>- north_padding: int32<br>- south_padding: int32<br>- padding_list: int32[] (default=[])<br>- pool_type: str<br>- pad_mode: str<br>- is_ceil_mode: bool (default=false)<br>- is_include_pad: bool (default=false)<br>- channel_last: bool (default=true) |
| GlobalAveragePooling |num inputs: 1<br>num outputs: 1<br>attributes:<br>- channel_last: bool (default=true) |
| GlobalResponseNormalization |num inputs: 1<br>num outputs: 1<br>attributes:<br>- scale: Weight<br>- bias: Weight<br>- epsilon: float32<br>- channel_last: bool (default=true) |
| Pow |num inputs: 1<br>num outputs: 1<br>attributes:<br>- exponent: float32 |
| QuickGelu |num inputs: 1<br>num outputs: 1<br> |
| ReduceL2 |num inputs: 1<br>num outputs: 1<br>attributes:<br>- keep_dims: bool<br>- reduce_axes: int32[]<br>- noop_with_empty_axes: int32 |
| ReduceMax |num inputs: 1<br>num outputs: 1<br>attributes:<br>- keep_dims: bool<br>- reduce_axes: int32[]<br>- noop_with_empty_axes: int32 |
| ReduceMin |num inputs: 1<br>num outputs: 1<br>attributes:<br>- keep_dims: bool<br>- reduce_axes: int32[]<br>- noop_with_empty_axes: int32 |
| ReduceMean |num inputs: 1<br>num outputs: 1<br>attributes:<br>- keep_dims: bool<br>- reduce_axes: int32[]<br>- noop_with_empty_axes: int32 |
| ReduceProd |num inputs: 1<br>num outputs: 1<br>attributes:<br>- keep_dims: bool<br>- reduce_axes: int32[]<br>- noop_with_empty_axes: int32 |
| ReduceSum |num inputs: 1<br>num outputs: 1<br>attributes:<br>- keep_dims: bool<br>- reduce_axes: int32[]<br>- noop_with_empty_axes: int32 |
| ReduceRMS |num inputs: 1<br>num outputs: 1<br>attributes:<br>- keep_dims: bool<br>- reduce_axes: int32[]<br>- epsilon: float32 (optional)<br><br>description:<br>reduce rms with epsilon if exists. i.e.,sqrt(mean**2 + eps)<br>we let eps to be optional so that it has more general fusing opportunities. |
| MeanCentering |num inputs: 1<br>num outputs: 1<br>attributes:<br>- axes: int32[]<br><br>description:<br>compute x - mean(x, axes) |
| Relu |num inputs: 1<br>num outputs: 1<br> |
| Reshape |num inputs: 1<br>num outputs: 1<br>attributes:<br>- new_shape: int32[] (optional)<br>- allowzero: int32 (default=0)<br>- is_const_first: bool (default=false) |
| ReshapeV2 |num inputs: 2<br>num outputs: 1<br>attributes:<br>- allowzero: int32 (default=0)<br>- is_const_first: bool (default=false)<br><br>description:<br>Reshape with variable shape argument. inputs[1] will be used as a shape. |
| Resize |num inputs: 1<br>num outputs: 1<br>attributes:<br>- resize_type: str (optional)<br>- resize_mode: str (optional)<br>- nearest_mode: str (optional)<br>- transform_mode: str (optional)<br>- roi: int32[] (optional)<br>- scales: float32[] (optional)<br>- sizes: int32[] (optional)<br>- cubic_coeff_a: float32 (optional)<br>- exclude_outside: int32 (optional)<br>- extrapolation_value: float32 (optional)<br>- antialias: int32 (optional)<br>- axes: int32[] (optional)<br>- keep_aspect_ratio_policy: str (optional) |
| RmsNormalization |num inputs: 1<br>num outputs: 1<br>attributes:<br>- normalized_shape: int64[] (default=[])<br>- epsilon: Weight<br>- scale: Weight<br>- rms_center: Weight (optional) |
| RotaryEmbedding |num inputs: 1<br>num outputs: 2<br>attributes:<br>- inv_freq: Weight<br>- attention_scaling: float32<br><br>description:<br>following the LlamaRotaryEmbedding of the Huggingface Transformer library.<br>inputs => (position_ids,), outputs => (cos, sin) |
| Rsqrt |num inputs: 1<br>num outputs: 1<br> |
| Rstd |num inputs: 1<br>num outputs: 1<br>attributes:<br>- reduce_axes: int32[]<br>- epsilon: Weight<br>- num_groups: int32 (default=1)<br><br>description:<br>computes 1/std(x). if gruops > 1, then activation is grouped along channel axis. |
| Gather |num inputs: 2<br>num outputs: 1<br>attributes:<br>- axis: int32 (default=0) |
| GatherElements |num inputs: 2<br>num outputs: 1<br>attributes:<br>- axis: int32 (default=0) |
| ScatterND |num inputs: variable<br>num outputs: 1<br>attributes:<br>- constant_data: Weight (optional)<br>- constant_indices: Weight (optional) |
| Selu |num inputs: 1<br>num outputs: 1<br> |
| Shape |num inputs: 1<br>num outputs: 1<br>attributes:<br>- start: int32 (optional)<br>- end: int32 (optional) |
| Sigmoid |num inputs: 1<br>num outputs: 1<br>attributes:<br>- alpha: float32 (default=1.0)<br><br>description:<br>alpha is added to deal with more general case like 1 / (1 + exp(-alpha * x)) |
| InverseSigmoid |num inputs: 1<br>num outputs: 1<br>attributes:<br>- eps: float32 (default=1e-05)<br><br>description:<br>x = x.clamp(x, min=0, max=1)<br>x1 = x.clamp(min=eps)<br>x2 = (1 - x).clamp(min=eps)<br>out = log(x1 / x2) |
| Slice |num inputs: 1<br>num outputs: 1<br>attributes:<br>- axis: int32[]<br>- start: int64[]<br>- end: int64[]<br>- step: int64[] (optional) |
| Softmax |num inputs: 1<br>num outputs: 1<br>attributes:<br>- beta: float32 (default=1.0)<br>- axis: int32 |
| LogSoftmax |num inputs: 1<br>num outputs: 1<br>attributes:<br>- axis: int32 |
| Softplus |num inputs: 1<br>num outputs: 1<br>attributes:<br>- beta: float32 (default=1.0)<br>- threshold: float32 (default=20.0) |
| Softsign |num inputs: 1<br>num outputs: 1<br> |
| Sqrt |num inputs: 1<br>num outputs: 1<br> |
| StaggeredPadding |num inputs: 1<br>num outputs: 1<br>attributes:<br>- axis: int32 (default=3)<br>- reverse: bool (default=false)<br>- window_size: int32 (default=-1)<br><br>description:<br>StaggeredPadding does the following:<br>[[[[1,1,1],<br>[2,2,2],<br>[3,3,3],<br>[4,4,4]]<br>[5,5,5]]]]<br>=> [[[[1,1,1,0,0,0,0],<br>[0,2,2,2,0,0,0],<br>[0,0,3,3,3,0,0],<br>[0,0,0,4,4,4,0],<br>[0,0,0,0,5,5,5]]]]<br>That is, each original row is shifted to the right so that its non-zero block lies on a “diagonal,” and all other elements are zero-padded.<br>Then slice last dim by (2*window_size + 1). For example, if window_size=1 in the above, 5=2*1+1 elements are sliced.<br>=> [[[[1,1,0,0,0],<br>[2,2,2,0,0],<br>[0,3,3,3,0],<br>[0,0,4,4,4],<br>[0,0,0,5,5]]]]<br>If reverse<br>[[[[1,1,1],<br>[2,2,2],<br>[3,3,3],<br>[4,4,4]]<br>[5,5,5]]]]<br>=> [[[[0,0,0,0,1,1,1],<br>[0,0,0,2,2,2,0],<br>[0,0,3,3,3,0,0],<br>[0,4,4,4,0,0,0],<br>[5,5,5,0,0,0,0]]]] |
| StatefulAttentionMaskWrapper |num inputs: 1 or 2<br>num outputs: 1<br>attributes:<br>- past_seqlen: int32 (default=0)<br>- is_sliding: bool (default=false)<br>- sliding_window: int32 (default=1024)<br>- mask_type: str (default="causal_mask")<br><br>description:<br>mask_type: "causal_mask", "custom_mask" — "custom_mask" takes the mask as a second input |
| StatefulKVCacheWrapper |num inputs: 1<br>num outputs: 1<br>attributes:<br>- cache_name: str<br>- past_seqlen: int32 (default=0)<br>- is_sliding: bool (default=false)<br>- sliding_window: int32 (default=1024)<br>- max_cache_len: int32 (default=8192) |
| StatefulCachedStack |num inputs: 1<br>num outputs: 1<br>attributes:<br>- axis: int32 (default=2)<br>- cache_len: int32 (default=0)<br>- cache_name: str |
| StatefulLSTMWrapper |num inputs: variable<br>num outputs: 3<br>attributes:<br>- W: Weight<br>- R: Weight<br>- B: Weight (optional)<br>- input_mask: str<br>- output_mask: str<br>- P: Weight (optional)<br>- direction: str (default="forward")<br>- batch_first: bool (default=false)<br>- hidden_size: int32<br>- hidden_state_name: str<br>- cell_state_name: str<br>- subgraph: str<br>- activation_alpha: float32[] (optional)<br>- activation_beta: float32[] (optional)<br>- activations: str[] (optional)<br>- clip: float32 (optional)<br>- input_forget: bool (default=false)<br><br>description:<br>When input_mask is `ishc`, then inputs are [input, sequence_lens, initial_h, initial_c],<br>and if input_mask is `ihc`, then inputs are [input, initial_h, initial_c].<br>The output_mask is `h` means that among [output, hidden_out, cell_out], only the hidden_out is being used. |
| StatefulRNNWrapper |num inputs: variable<br>num outputs: 2<br>attributes:<br>- W: Weight<br>- R: Weight<br>- B: Weight (optional)<br>- input_mask: str<br>- output_mask: str<br>- direction: str (default="forward")<br>- batch_first: bool (default=false)<br>- hidden_size: int32<br>- hidden_state_name: str<br>- subgraph: str<br>- activation_alpha: float32[] (optional)<br>- activation_beta: float32[] (optional)<br>- activations: str[] (optional)<br>- clip: float32 (optional)<br><br>description:<br>When input_mask is `ish`, then inputs are [input, sequence_lens, initial_h],<br>and if input_mask is `ih`, then inputs are [input, initial_h].<br>The output_mask is `h` means that among [output, hidden_out], only the hidden_out is being used. |
| StatefulGRUWrapper |num inputs: variable<br>num outputs: 2<br>attributes:<br>- W: Weight<br>- R: Weight<br>- B: Weight (optional)<br>- input_mask: str<br>- output_mask: str<br>- direction: str (default="forward")<br>- batch_first: bool (default=false)<br>- hidden_size: int32<br>- hidden_state_name: str<br>- subgraph: str<br>- activation_alpha: float32[] (optional)<br>- activation_beta: float32[] (optional)<br>- activations: str[] (optional)<br>- clip: float32 (optional)<br>- linear_before_reset: bool (default=false)<br><br>description:<br>When input_mask is `ish`, then inputs are [input, sequence_lens, initial_h],<br>and if input_mask is `ih`, then inputs are [input, initial_h].<br>The output_mask is `h` means that among [output, hidden_out], only the hidden_out is being used. |
| StatefulRoPEWrapper |num inputs: 3<br>num outputs: 1<br>attributes:<br>- rope_type: int32 (default=0)<br>- seqlen: int32 (default=0)<br>- rotary_emb_dim: int32 (default=0)<br><br>description:<br>StatefulRoPEWrapper<br>inputs: (q_or_k, con, sin)<br>outputs: (q_or_k_embed)<br>rope_type=> 0: llama, 1: cohere2(aya-vision) |
| SubConstant |num inputs: 1<br>num outputs: 1<br>attributes:<br>- constant: Weight<br>- is_const_first: bool<br>- shape: int32[] (default=[]) |
| Sub |num inputs: 2<br>num outputs: 1<br> |
| Swish |num inputs: 1<br>num outputs: 1<br> |
| Tanh |num inputs: 1<br>num outputs: 1<br> |
| Sin |num inputs: 1<br>num outputs: 1<br> |
| Cos |num inputs: 1<br>num outputs: 1<br> |
| TransposeConvolution |num inputs: 1<br>num outputs: 1<br>attributes:<br>- in_channels: int32<br>- out_channels: int32<br>- h_kernel: int32<br>- w_kernel: int32<br>- h_stride: int32<br>- w_stride: int32<br>- h_dilation: int32<br>- w_dilation: int32<br>- west_padding: int32<br>- east_padding: int32<br>- north_padding: int32<br>- south_padding: int32<br>- padding_list: int32[] (default=[])<br>- og_padding_list: int32[] (default=[])<br>- og_out_padding_list: int32[] (default=[])<br>- group_number: int32<br>- weight: Weight<br>- bias: Weight (optional)<br>- channel_last: bool (default=true) |
| Transpose |num inputs: 1<br>num outputs: 1<br>attributes:<br>- transpose_axes: int64[] |
| Unsqueeze |num inputs: 1<br>num outputs: 1<br>attributes:<br>- axes: int32[] |
| OrderedPatchify |num inputs: 1<br>num outputs: 1<br>attributes:<br>- h_kernel: int32<br>- w_kernel: int32<br>- order: int32 (default=0)<br><br>description:<br>This operation splits a 4D input tensor into non-overlapping patches and rearranges them into a sequence<br>according to the specified order. Supports row-major (order=0) and column-major (order=1) layouts.<br>If order=0 (row-major), the equivalent is: 256 patches, each 12x12, (h_kernel=w_kernel=12)<br>(1,192,192,64)-[reshape]->(1,16,12,16,12,64)-[transpose 013245]->(1,16,16,12,12,64)-[reshape]->(1,256,144,64)<br>Note that order0 is its own inverse.<br>If order=1 (column-major), the equivalent is: 144 patches, each 16x16, (h_kernel=w_kernel=12)<br>(1,192,192,64)-[reshape]->(1,16,12,16,12,64)-[transpose 024135]->(1,12,12,16,16,64)-[reshape]->(1,144,256,64)<br>if order=2 (reversing order=1), h_kernel=12, w_kernel=16<br>(1,144,256,64)-[reshape]->(1,12,12,16,16,64)-[transpose 031425]->(1,16,12,16,12,64)-[reshape]->(1,192,192,64) |
| DynamicSliceInputConst |num inputs: 1<br>num outputs: 1<br>attributes:<br>- mod: int32<br>- start: int64[]<br>- end: int64[]<br>- axis: int64[]<br>- step: int64[] (optional) |
| RelShift |num inputs: 1<br>num outputs: 1<br> |
| MeloTTSPiecewiseRationalQuadraticTransform |num inputs: 2<br>num outputs: 1<br>attributes:<br>- num_bins: int32<br>- filter_channels: int32<br>- tail_bound: float32<br>- reverse: bool<br><br>description:<br>helper function for parsing melotts model. |
| Neg |num inputs: 1<br>num outputs: 1<br> |
| Einsum |num inputs: variable<br>num outputs: 1<br>attributes:<br>- equation: str<br>- constant: Weight (optional)<br>- is_const_first: bool (default=false) |
| Where |num inputs: 3<br>num outputs: 1<br><br><br>description:<br>condition,X,Y, if cond == True: else  y |
| TensorCmp |num inputs: 2<br>num outputs: 1<br>attributes:<br>- kind: str (default="eq")<br><br>description:<br>kind: eq, neq, lt, le, gt, ge |
| Logical |num inputs: variable<br>num outputs: 1<br>attributes:<br>- kind: str (default="and")<br><br>description:<br>this function represents the logical operation of two inputs.<br>single input if kind is "not" else two inputs<br>kind: and, or, xor, not |
| MaskedFill |num inputs: 2<br>num outputs: 1<br>attributes:<br>- value: float32<br><br>description:<br>inputs: (x, mask) |
| SubgraphCall |num inputs: variable<br>num outputs: variable<br>attributes:<br>- subgraph: str<br>- scope: str<br>- kind: str |
| MoeDispatcher |num inputs: 2<br>num outputs: 2<br>attributes:<br>- top_k: int32<br>- norm_top_k_prob: bool (default=true)<br><br>description:<br>inputs: (hidden_states, router_logits)<br>outputs: (hidden_states, top_k_logits)<br>if norm_top_k_prob is true<br>topk_vals, topk_idx = torch.topk(<br>routing_weights, self.top_k, dim=-1<br>)  # (18,8), (18,8)<br>weight_new = torch.zeros_like(routing_weights)<br>weight_new.scatter_(1, topk_idx, topk_vals)<br>routing_weights = weight_new<br>routing_weights /= routing_weights.sum(dim=1, keepdim=True) |
| MoeAggregator |num inputs: variable<br>num outputs: 1<br><br><br>description:<br>inputs: (router_weights, experts_output0, expert_output1, ...) |
| TopKValues |num inputs: 1<br>num outputs: 1<br>attributes:<br>- axis: int32<br>- k: int32 (default=1)<br>- largest: bool (default=true)<br>- sorted: bool (default=true) |
| TopKIndices |num inputs: 1<br>num outputs: 1<br>attributes:<br>- axis: int32<br>- k: int32 (default=1)<br>- largest: bool (default=true)<br>- sorted: bool (default=true) |
