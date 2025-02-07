/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "tl_ucp_sendrecv.h"

/* TODO: add as parameters */
#define MSG_MEDIUM 66000
#define NP_THRESH 32

static inline ucc_rank_t get_recv_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank + step) % size;
}

static inline ucc_rank_t get_send_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank - step + size) % size;
}

#if 0
static ucc_rank_t get_num_posts(const ucc_tl_ucp_team_t *team,
                                const ucc_coll_args_t *args)
{
    unsigned long posts = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_num_posts;
    ucc_rank_t    tsize = UCC_TL_TEAM_SIZE(team);
    size_t data_size;

    data_size = (size_t)args->src.info.count *
                ucc_dt_size(args->src.info.datatype);
    if (posts == UCC_ULUNITS_AUTO) {
        if ((data_size > MSG_MEDIUM) && (tsize > NP_THRESH)) {
            /* use pairwise algorithm */
            posts = 1;
        } else {
            /* use linear algorithm */
            posts = 0;
        }
    }

    posts = (posts > tsize || posts == 0) ? tsize: posts;
    return posts;
}
#endif

static ucc_rank_t get_num_tokens(const ucc_tl_ucp_team_t *team,
                                 const ucc_coll_args_t *args)
{
    unsigned long tokens;
    ucc_rank_t    grank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t    gsize = UCC_TL_TEAM_SIZE(team);
    size_t        data_size_per_process;
    int           ppn = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_ca_ppn;
    unsigned long data_transfer_ceiling = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_ca_rate;
    unsigned long data_size;

    //data_size_per_process = (size_t)(TASK_ARGS(task).src.info.count / gsize) *
    //                        ucc_dt_size(TASK_ARGS(task).src.info.datatype);

    data_size_per_process = (size_t)(args->src.info.count / gsize) *
                            ucc_dt_size(args->src.info.datatype);

    // amount of incoming/outgoing data (in bytes) per node
    // assuming message to self with not use network bw
    data_size = data_size_per_process * (gsize - 1) * ppn; 

    //converting MBs to Bs first
    tokens = (data_transfer_ceiling * 1000 * 1000) / data_size;

    int calc_tokens = tokens;

    tokens = (tokens > gsize) ? gsize: tokens;

    if (tokens < 1) 
        tokens = 1;

    if (grank == 0)
        tl_debug(UCC_TL_TEAM_LIB(team), "net_bw=%ld, data_size =%ld, ppn=%d, tokens=%ld, calc_tokens=%d \n", 
		         (data_transfer_ceiling * 1000 * 1000), data_size, ppn, tokens, calc_tokens);

    return tokens;
}

void ucc_tl_ucp_alltoall_pairwise_ca_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ptrdiff_t          sbuf  = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  smem  = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t  rmem  = TASK_ARGS(task).dst.info.mem_type;
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize = UCC_TL_TEAM_SIZE(team);
    int                polls = 0;
    ucc_rank_t         peer, nreqs;
    size_t             data_size;

    nreqs     = get_num_tokens(team, &TASK_ARGS(task));

    data_size = (size_t)(TASK_ARGS(task).src.info.count / gsize) *
                ucc_dt_size(TASK_ARGS(task).src.info.datatype);

    while ((task->tagged.send_posted < gsize ||
            task->tagged.recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);
        while ((task->tagged.recv_posted < gsize) &&
               ((task->tagged.recv_posted - task->tagged.recv_completed) <
                nreqs)) {
            peer = get_recv_peer(grank, gsize, task->tagged.recv_posted);
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb((void *)(rbuf + peer * data_size),
                                             data_size, rmem, peer, team, task),
                                             task, out);
	    if (grank == 0)
                tl_debug(UCC_TL_TEAM_LIB(team), "%d of %d: peer=%d, nreqs=%d, "
	    	         "data_size=%ld, count=%ld\n", grank, gsize, peer, nreqs, 
		          data_size, TASK_ARGS(task).src.info.count);
            polls = 0;
        }
        while ((task->tagged.send_posted < gsize) &&
               ((task->tagged.send_posted - task->tagged.send_completed) <
                nreqs)) {
            peer = get_send_peer(grank, gsize, task->tagged.send_posted);
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb((void *)(sbuf + peer * data_size),
                                             data_size, smem, peer, team, task),
                          task, out);
            polls = 0;
        }
    }
    if ((task->tagged.send_posted < gsize) ||
        (task->tagged.recv_posted < gsize)) {
        return;
    }

    task->super.status = ucc_tl_ucp_test(task);
out:
    if (task->super.status != UCC_INPROGRESS) {
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                         "ucp_alltoall_pairwise_ca_done", 0);
    }
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_ca_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_alltoall_pairwise_ca_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_ca_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    size_t data_size;

    task->super.post     = ucc_tl_ucp_alltoall_pairwise_ca_start;
    task->super.progress = ucc_tl_ucp_alltoall_pairwise_ca_progress;

    task->n_polls = ucc_max(1, task->n_polls);
    if (UCC_TL_UCP_TEAM_CTX(team)->cfg.pre_reg_mem) {
        data_size =
            (size_t)args->src.info.count * ucc_dt_size(args->src.info.datatype);
        ucc_tl_ucp_pre_register_mem(team, args->src.info.buffer, data_size,
                                    args->src.info.mem_type);
        ucc_tl_ucp_pre_register_mem(team, args->dst.info.buffer, data_size,
                                    args->dst.info.mem_type);
    }

    return UCC_OK;
}
