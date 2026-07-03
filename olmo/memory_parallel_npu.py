import torch
import torch.distributed as dist
import torch_npu

_MEMORY_LAYER_PARALLEL_GROUP = None
_MEMORY_LAYER_DATA_PARALLEL_GROUP = None
_MPU_MEMORY_LAYER_DATA_PARALLEL_RANK= None
_MPU_MEMORY_LAYER_PARALLEL_RANK = None
_MPU_MEMORY_LAYER_PARALLEL_WORLD_SIZE = None
_MPU_MEMORY_LAYER_DATA_PARALLEL_WORLD_SIZE = None

def init_memory_parallel_group(memory_layer_parallel_size):
    if memory_layer_parallel_size == 1:
        return
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"DEBUG [Rank {rank}]: Initializing Memory Parallel. World: {world_size}, MP Size: {memory_layer_parallel_size}")
    # Build the memory layer parallel groups.
    global _MEMORY_LAYER_PARALLEL_GROUP
    assert _MEMORY_LAYER_PARALLEL_GROUP is None, \
        'Memory layer parallel group is already initialized'
    assert world_size % memory_layer_parallel_size == 0
    memory_layer_dp_size = world_size // memory_layer_parallel_size
    for i in range(memory_layer_dp_size):
        ranks = range(i * memory_layer_parallel_size,
                      (i + 1) * memory_layer_parallel_size)
        group = dist.new_group(ranks, backend="hccl")
        if rank in ranks:
            print(f"DEBUG [Rank {rank}]: Setting global group. Ranks: {list(ranks)}")
            _MEMORY_LAYER_PARALLEL_GROUP = group

    global _MEMORY_LAYER_DATA_PARALLEL_GROUP
    assert _MEMORY_LAYER_DATA_PARALLEL_GROUP is None, \
        'Memory layer data parallel group is already initialized'
    for i in range(memory_layer_parallel_size):
        start_rank = i
        end_rank = memory_layer_dp_size * memory_layer_parallel_size + i
        ranks = range(start_rank, end_rank, memory_layer_parallel_size)
        group = dist.new_group(ranks, backend="hccl")
        if rank in ranks:
            _MEMORY_LAYER_DATA_PARALLEL_GROUP = group

def get_memory_layer_parallel_group():
    """Get the memory layer parallel group the caller rank belongs to."""
    assert _MEMORY_LAYER_PARALLEL_GROUP is not None, \
        'memory layer parallel group is not initialized'
    return _MEMORY_LAYER_PARALLEL_GROUP

def get_memory_layer_data_parallel_group():
    """Get the memory layer parallel group the caller rank belongs to."""
    assert _MEMORY_LAYER_DATA_PARALLEL_GROUP is not None, \
        'memory layer parallel group is not initialized'
    return _MEMORY_LAYER_DATA_PARALLEL_GROUP

def get_memory_layer_parallel_rank():
    """Return my rank for the memory layer parallel group."""
    global _MPU_MEMORY_LAYER_PARALLEL_RANK
    if _MPU_MEMORY_LAYER_PARALLEL_RANK is not None:
        return _MPU_MEMORY_LAYER_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_memory_layer_parallel_group())

def get_memory_layer_data_parallel_rank():
    """Return my rank for the memory layer parallel group."""
    global _MPU_MEMORY_LAYER_DATA_PARALLEL_RANK
    if _MPU_MEMORY_LAYER_DATA_PARALLEL_RANK is not None:
        return _MPU_MEMORY_LAYER_DATA_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_memory_layer_data_parallel_group())

def get_memory_layer_parallel_world_size():
    """Return world size for the memory layer parallel group."""
    global _MPU_MEMORY_LAYER_PARALLEL_WORLD_SIZE
    if _MPU_MEMORY_LAYER_PARALLEL_WORLD_SIZE is not None:
        return _MPU_MEMORY_LAYER_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_memory_layer_parallel_group())

def get_memory_layer_data_parallel_world_size():
    """Return world size for the memory layer data parallel group."""
    global _MPU_MEMORY_LAYER_DATA_PARALLEL_WORLD_SIZE
    if _MPU_MEMORY_LAYER_DATA_PARALLEL_WORLD_SIZE is not None:
        return _MPU_MEMORY_LAYER_DATA_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_memory_layer_data_parallel_group())

def _gather_along_first_dim(input_):
    dim_size = list(input_.size())
    world_size = get_memory_layer_parallel_world_size()
    if world_size == 1:
        return input_
    dim_size[0] = dim_size[0] * world_size
    group = get_memory_layer_parallel_group()
    output = torch.empty(dim_size, dtype=input_.dtype,
                         #device=torch.cuda.current_device())
                         device=torch.npu.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(),
                                    group=group)
    return output

def _reduce_scatter_along_first_dim(input_):
    world_size = get_memory_layer_parallel_world_size()
    if world_size == 1:
        return input_
    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         #device=torch.cuda.current_device())
                         device=torch.npu.current_device())
    group = get_memory_layer_parallel_group()
    torch.distributed._reduce_scatter_base(output, input_.contiguous(),
                                           group=group)
    return output


class AllReduceInMemGroup(torch.autograd.Function):
    """Gather the input from memory region and concatinate."""

    @staticmethod
    def forward(ctx, input_):
        group = get_memory_layer_parallel_group()
        torch.distributed.all_reduce(input_, group=group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        group = get_memory_layer_parallel_group()
        torch.distributed.all_reduce(grad_output, group=group)
        return grad_output

class _GatherFromDplRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_first_dim(grad_output)

def gather_from_memory_layer_parallel_region(input_):
    return _GatherFromDplRegion.apply(input_)



class MultiHeadGatherScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *scores):
        cur_rank = torch.distributed.get_rank()
        num_head = len(scores)
        ctx.num_head = num_head
        mem_layer_parallel_size = get_memory_layer_parallel_world_size()
        memory_sub_group_size = mem_layer_parallel_size // num_head
        memory_src_rank = cur_rank // mem_layer_parallel_size * mem_layer_parallel_size
        output = torch.empty(mem_layer_parallel_size * scores[0].numel(), device=scores[0].device, dtype=scores[0].dtype, requires_grad=True)
        ctx.score_shape = scores[0].shape

        out_list=[]
        for i in range(len(scores)):
            dst_rank = memory_src_rank + i * memory_sub_group_size
            rank_range = range(dst_rank, dst_rank + memory_sub_group_size)
            if torch.distributed.get_rank() in rank_range:
                sub_group_id=i
            out_list.append(_gather_along_first_dim(scores[i]))
        output = out_list[sub_group_id]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mem_layer_parallel_size = get_memory_layer_parallel_world_size()
        cur_rank = torch.distributed.get_rank()
        memory_sub_group_size = mem_layer_parallel_size // ctx.num_head
        memory_src_rank = cur_rank // mem_layer_parallel_size * mem_layer_parallel_size

        scores_grad_list = []
        for i in range(ctx.num_head):
            dst_rank = memory_src_rank + i * memory_sub_group_size
            rank_range = range(dst_rank, dst_rank + memory_sub_group_size)
            if torch.distributed.get_rank() in rank_range:
                out_grad = grad_output
            else:
                out_grad = torch.zeros_like(grad_output)
            scores_grad_list.append(_reduce_scatter_along_first_dim(out_grad))

        return tuple(scores_grad_list)


def tucker_multihead_gather_score(*scores):
    return MultiHeadGatherScore().apply(*scores)


class EmbeddingAll2AllSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        """_summary_

        Args:
            count_send (list[int]): input element count for each rank
            count_receive(list[int]): receive element count for each rank
        """
        all2all_input, all2all_output, ctx.count_send, ctx.count_receive  = inputs
        ctx.input_shape = all2all_input.shape
        # print(all2all_inputs)
        torch.distributed.all_to_all_single(all2all_output,
                                    all2all_input,
                                    output_split_sizes=ctx.count_receive,
                                    input_split_sizes=ctx.count_send,
                                    group=get_memory_layer_parallel_group())
        return all2all_output


    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        grad_input = torch.empty(input_shape, dtype=grad_output.dtype, device=grad_output.device)
        torch.distributed.all_to_all_single(
                                    grad_input,
                                    grad_output.contiguous(),
                                    output_split_sizes=ctx.count_send,
                                    input_split_sizes=ctx.count_receive,
                                    group=get_memory_layer_parallel_group())
        return (grad_input, None, None, None)
