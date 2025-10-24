"""GPU Batched, Tensor-based IK for Spot Robot Motion Skills."""

import math

import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_quaternion,
)
from mani_skill.utils.structs.pose import Pose

SHOULDER_OFFSET = np.linalg.inv(
    np.array([[1, 0, 0, 0.292], [0, 1, 0, 0], [0, 0, 1, 0.188], [0, 0, 0, 1]])
)

HAND2WRIST_POSE = Pose.create_from_pq(
    torch.tensor([-0.1955707, 0, 0]), torch.tensor([1.0, 0.0, 0.0, 0.0])
)

#############################
# Batched 2R IK for PyTorch
################################


def IK2R_batched(
    L1: float, L2: float, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched 2R planar IK for link lengths L1, L2.

    Args:
        L1, L2: link lengths
        x, y: torch.Tensor of shape (B,), target end-effector coordinates in plane.
    Returns:
        sols: torch.Tensor of shape (B, 2, 2), each entry [i, j] = (q2, q3) solution j for sample i.
        valid: torch.BoolTensor of shape (B,) indicating if c2 in [-1,1].
    """
    # Law of cosines
    xy2 = x**2 + y**2
    c2 = (xy2 - L1**2 - L2**2) / (2 * L1 * L2)
    valid = c2.abs() <= 1.0
    c2_clamped = c2.clamp(-1.0, 1.0)

    # two elbow angles
    q3a = torch.acos(c2_clamped)
    q3b = -q3a

    # shoulder base angle
    theta = torch.atan2(y, x)
    alpha_a = torch.atan2(L2 * torch.sin(q3a), L1 + L2 * torch.cos(q3a))
    alpha_b = torch.atan2(L2 * torch.sin(q3b), L1 + L2 * torch.cos(q3b))

    q2a = theta - alpha_a
    q2b = theta - alpha_b

    # stack solutions: (B, 2, 2) dims: sample, solution-index, [q2, q3]
    sols = torch.stack(
        [torch.stack([q2a, q3a], dim=1), torch.stack([q2b, q3b], dim=1)], dim=1
    )
    return sols, valid


def analytic_spot_ik_6_torch(
    wrist_pose: torch.Tensor,  # (B,4,4)
    min_limits: torch.Tensor,  # (6,)
    max_limits: torch.Tensor,  # (6,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched analytic IK for Spot 6-DOF arm.

    Args:
        wrist_pose: (B,4,4) transform wrist->shoulder
        min_limits, max_limits: (6,) joint bounds
    Returns:
        solutions: torch.Tensor of shape (B, 8, 6)
        valid: torch.Tensor of shape (B, 8) mask of solutions within limits
    """
    B = wrist_pose.shape[0]
    # link lengths
    l2 = 0.3385
    l3 = (0.40330**2 + 0.0750**2) ** 0.5
    q3_off = torch.atan2(torch.tensor(0.0750), torch.tensor(0.40330))

    # extract wrist pos
    px = wrist_pose[:, 0, 3]
    py = wrist_pose[:, 1, 3]
    pz = wrist_pose[:, 2, 3]
    xl = torch.sqrt(px**2 + py**2)

    # first planar IK
    sols1, _ = IK2R_batched(l2, l3, xl, -pz)
    # second (rotated) planar IK
    sols2, _ = IK2R_batched(l2, l3, -xl, -pz)

    # build q1 for each sample
    q1_base = torch.atan2(py, px)  # (B,)
    q1_1 = q1_base.unsqueeze(1).expand(-1, 2)  # (B,2)
    q1_2 = (q1_base + math.pi).unsqueeze(1).expand(-1, 2)

    # assemble shoulder/elbow solutions: (B,4,3)
    sols1[:, :, 1] += q3_off  # adjust q3 by offset
    sols2[:, :, 1] += q3_off  # adjust q3 by offset
    q2q3_1 = sols1  # (B,2,2)
    q2q3_2 = sols2  # (B,2,2)
    sol123 = torch.cat(
        [
            torch.cat([q1_1.unsqueeze(2), q2q3_1], dim=2),
            torch.cat([q1_2.unsqueeze(2), q2q3_2], dim=2),
        ],
        dim=1,
    )  # (B,4,3)

    # expand for batch of solutions
    BS = B * 4
    sol123_bs = sol123.reshape(BS, 3)
    # prepare wrist poses per solution
    wrist_bs = wrist_pose.unsqueeze(1).expand(-1, 4, -1, -1).reshape(BS, 4, 4)

    # build T_r3 (BS,4,4)
    q1_bs = sol123_bs[:, 0]
    q23_sum = sol123_bs[:, 1] + sol123_bs[:, 2]

    # rotation matrices
    def rot_z(q: torch.Tensor) -> torch.Tensor:
        """Rotation matrix around Z-axis for angle q."""
        c = torch.cos(q)
        s = torch.sin(q)
        zeros = torch.zeros_like(q)
        ones = torch.ones_like(q)
        R = torch.stack(
            [
                torch.stack([c, -s, zeros], dim=1),
                torch.stack([s, c, zeros], dim=1),
                torch.stack([zeros, zeros, ones], dim=1),
            ],
            dim=1,
        )
        return R

    def rot_y(q: torch.Tensor) -> torch.Tensor:
        """Rotation matrix around Y-axis for angle q."""
        c = torch.cos(q)
        s = torch.sin(q)
        zeros = torch.zeros_like(q)
        ones = torch.ones_like(q)
        R = torch.stack(
            [
                torch.stack([c, zeros, s], dim=1),
                torch.stack([zeros, ones, zeros], dim=1),
                torch.stack([-s, zeros, c], dim=1),
            ],
            dim=1,
        )
        return R

    Rz = rot_z(q1_bs)  # (BS,3,3)
    Ry = rot_y(q23_sum)
    # make homogeneous
    T_r3 = torch.eye(4, device=wrist_bs.device).unsqueeze(0).repeat(BS, 1, 1)
    T_r3[:, :3, :3] = Rz @ Ry
    # invert
    T_r3_inv = torch.inverse(T_r3)

    # compute W = T_r3_inv @ wrist_bs
    W = T_r3_inv @ wrist_bs  # (BS,4,4)

    # extract needed
    W00 = W[:, 0, 0]
    W10 = W[:, 1, 0]
    W20 = W[:, 2, 0]
    W01 = W[:, 0, 1]
    W02 = W[:, 0, 2]

    # two wrist solutions
    q5a = torch.acos(W00)
    q5b = -q5a
    q5 = torch.stack([q5a, q5b], dim=1)  # (BS,2)
    s5 = torch.sin(q5)

    q4 = torch.atan2(W10.unsqueeze(1) / s5, -W20.unsqueeze(1) / s5)
    q6 = torch.atan2(W01.unsqueeze(1) / s5, W02.unsqueeze(1) / s5)

    # stack full solutions: (BS,2,6)
    sol123_expand = sol123_bs.unsqueeze(1).expand(-1, 2, -1)  # (BS,2,3)
    sol_ws = torch.stack([q4, q5, q6], dim=2)  # (BS,2,3)
    sol_full_bs = torch.cat([sol123_expand, sol_ws], dim=2)  # (BS,2,6)

    # reshape back: (B,4,2,6) -> (B,8,6)
    sol_full = sol_full_bs.view(B, 8, 6)

    # mask within limits
    lo = min_limits.view(1, 1, -1)
    hi = max_limits.view(1, 1, -1)
    valid = (sol_full >= lo) & (sol_full <= hi)
    valid = valid.all(dim=2)  # (B,8)

    return sol_full, valid


def angle_diff_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the angle difference between two tensors of angles."""
    two_pi = 2 * torch.pi
    z = (x - y) % two_pi
    return torch.where(z > torch.pi, z - two_pi, z)


def get_l1_distance_torch(sols1: torch.Tensor, sols2: torch.Tensor) -> torch.Tensor:
    """Compute the L1 distance between two sets of joint solutions."""
    diff = angle_diff_torch(sols1, sols2).abs()
    return diff.max(dim=-1).values


def select_solution_torch(
    solutions: torch.Tensor,  # (B, N, 6)
    curr_joint_positions: torch.Tensor,  # (B, 6)
    valid_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:
    """Select the best IK solution per batch based on smallest max-angle difference to
    `curr_joint_positions`. Returns tensor of shape (B, 6).

    After selecting the tentative best, if any joint is ≈±π, we flip that joint (π->-π
    or -π->π) and keep the flip only if it reduces the max-angle error.
    """
    B = solutions.shape[0]

    # 1) Compute angle diffs and max-distance per solution
    diff = angle_diff_torch(
        solutions, curr_joint_positions.unsqueeze(1)
    ).abs()  # (B, N, 6)
    dist = diff.max(dim=2).values  # (B, N)
    if valid_mask is not None:
        inf = torch.full_like(dist, float("inf"))
        dist = torch.where(valid_mask, dist, inf)

    # 2) Pick the tentative best solution
    idx = dist.argmin(dim=1)  # (B,)
    batch_idx = torch.arange(B, device=solutions.device)
    tentative = solutions[batch_idx, idx]  # (B, 6)

    # 3) Detect joints near ±π
    # pi_thresh = 1e-3
    # pi = torch.tensor(math.pi, device=solutions.device)
    # near_pos_pi = (tentative - pi).abs() < pi_thresh    # (B, 6)
    # near_neg_pi = (tentative + pi).abs() < pi_thresh    # (B, 6)
    # flip_mask = near_pos_pi | near_neg_pi              # (B, 6)

    # # 4) If any joint needs flipping, build a flipped candidate
    # if flip_mask.any():
    #     flipped = tentative.clone()
    #     flipped[flip_mask] = -flipped[flip_mask]       # flip only the ±π entries

    #     # 5) Compute max-angle error for tentative vs. flipped
    #     orig_diff = (tentative - curr_joint_positions).abs()  # (B, 6)
    #     orig_dist = orig_diff.max(dim=1).values                             # (B,)

    #     flip_diff = (flipped - curr_joint_positions).abs()
    #     flip_dist = flip_diff.max(dim=1).values                             # (B,)

    #     # 6) For each batch, if flipped is strictly better, choose it
    #     use_flip = flip_dist < orig_dist                                    # (B,)
    #     if use_flip.any():
    #         tentative[use_flip] = flipped[use_flip]

    return tentative


def slerp_torch(
    q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Spherical linear interpolation (SLERP) between two batches of unit quaternions.

    Args:
        q0: Tensor of shape (..., 4), start quaternions (must be normalized).
        q1: Tensor of shape (..., 4), end quaternions (must be normalized).
        t:  Tensor of shape (...) with interpolation factors in [0, 1].
        eps: small threshold to fall back to lerp when angles are very small.

    Returns:
        Tensor of shape (..., 4), interpolated unit quaternions.
    """
    # ensure same shape for elementwise ops
    t = t.unsqueeze(-1) if t.dim() + 1 == q0.dim() else t

    # Compute cosine between q0 and q1, shape (..., 1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # Flip to take shortest path
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot)

    # Decide between lerp and slerp
    DOT_THRESH = 1.0 - eps
    use_lerp = dot > DOT_THRESH  # boolean mask shape (..., 1)

    # LERP + normalize fallback
    lerp = q0 + t * (q1 - q0)
    lerp = lerp / lerp.norm(dim=-1, keepdim=True)

    # Standard SLERP
    theta_0 = torch.acos(dot)  # angle between
    sin_0 = torch.sin(theta_0)
    a = torch.sin((1.0 - t) * theta_0) / sin_0
    b = torch.sin(t * theta_0) / sin_0
    slerp = a * q0 + b * q1

    # Combine results
    return torch.where(use_lerp, lerp, slerp)


def concatenate_matrices(*mats: torch.Tensor) -> torch.Tensor:
    """Return the matrix product of a sequence of transformation matrices, batched. If
    no matrices are given, returns a 4×4 identity.

    Args:
        *mats: Tensors of shape (..., 4, 4).  Batch-shapes must all be broadcastable.

    Returns:
        Tensor of shape (..., 4, 4) = mats[0] @ mats[1] @ ... @ mats[-1].

    Examples:
        >>> M = torch.rand(4, 4) - 0.5
        >>> torch.allclose(M, concatenate_matrices_torch(M))
        True
        >>> torch.allclose(M @ M.T, concatenate_matrices_torch(M, M.T))
        True

        # batched example
        >>> A = torch.eye(4).unsqueeze(0).expand(5, 4, 4)
        >>> B = torch.rand(5, 4, 4)
        >>> C = torch.rand(5, 4, 4)
        >>> out = concatenate_matrices_torch(A, B, C)
        >>> torch.allclose(out, A @ B @ C)
        True
    """
    if len(mats) == 0:
        return torch.eye(4)

    # determine the common batch shape by broadcasting all batch dims
    # take the batch shape of the first matrix
    batch_shape = mats[0].shape[:-2]
    dtype = mats[0].dtype
    device = mats[0].device

    # start from batched identity
    identity = torch.eye(4, dtype=dtype, device=device)
    if batch_shape:
        identity = identity.view((1, 4, 4)).expand(*batch_shape, 4, 4)

    result = identity
    for M in mats:
        result = result.matmul(M)
    return result


class SpotMotion:
    """Toolkit for Spot IK based motion skills."""

    def __init__(
        self,
        device: torch.device,
        gripper_scale: float = 0.002,
    ):
        """Initialize SpotMotion with joint limits.

        Args:
            min_limits: Tensor of shape (7,) with minimum joint limits.
            max_limits: Tensor of shape (7,) with maximum joint limits.
        """
        self.min_limits = torch.as_tensor(
            [-2.6179938, -3.1415927, 0.0, -2.7925267, -1.8325957, -2.8797932],
            dtype=torch.float32,
            device=device,
        )
        self.max_limits = torch.as_tensor(
            [3.1415927, 0.5235988, 3.1415927, 2.7925267, 1.8325957, 2.8797932],
            dtype=torch.float32,
            device=device,
        )
        self.gripper_closed = 0.0  # for spot, larger finger angle is closing
        self.gripper_open = -1.57
        self.gripper_closing_delta = (
            self.gripper_closed - self.gripper_open
        ) * gripper_scale
        self.gripper_openning_delta = -self.gripper_closing_delta
        self.device = device
        self.hand2wrist_pose = HAND2WRIST_POSE.to(device)
        self.shoulder_offset = torch.tensor(
            SHOULDER_OFFSET, dtype=torch.float32, device=device
        )

    def solve_spot_ik(
        self,
        base_pose: Pose,
        end_effector_pose: Pose,
        curr_joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Solve Spot arm IK for a given wrist pose.

        Returns a joint configuration array of shape (8, 6). For now, use batch = 1, but
        this can be batched.
        """
        B = curr_joint_positions.shape[0]
        hand2wrist_pose = Pose.create_from_pq(
            self.hand2wrist_pose.p.repeat(B, 1), self.hand2wrist_pose.q.repeat(B, 1)
        )
        wrist_pose_worldF = end_effector_pose * hand2wrist_pose
        wrist_pose_robotF = base_pose.inv() * wrist_pose_worldF
        wrist_pose_robotF_mat = wrist_pose_robotF.to_transformation_matrix()
        shoulder_offset = self.shoulder_offset.unsqueeze(0).repeat(B, 1, 1)
        wrist_pose_shoulderF = concatenate_matrices(
            shoulder_offset, wrist_pose_robotF_mat
        )

        solutions, valid = analytic_spot_ik_6_torch(
            wrist_pose_shoulderF, self.min_limits, self.max_limits
        )
        selected_solution = select_solution_torch(
            solutions, curr_joint_positions, valid
        )

        return selected_solution

    def build_grasp_pose(
        self, approaching: torch.Tensor, closing: torch.Tensor, center: torch.Tensor
    ) -> Pose:
        """Build a grasp pose (spot_hand_frame)."""
        # assert (torch.abs(1 - torch.norm(approaching, dim=-1)) < 1e-3).all()
        # assert (torch.abs(1 - torch.norm(closing, dim=-1)) < 1e-3).all()
        # assert (
        #     torch.bmm(approaching.unsqueeze(1), closing.unsqueeze(-1)) <= 5e-3
        # ).all()
        B = approaching.shape[0]
        ortho = torch.cross(approaching, closing)  # x cross y = z
        T = torch.stack([approaching, closing, ortho], dim=2)
        q = matrix_to_quaternion(T)
        overlapping = Pose.create_from_pq(center, q)
        relative_pose = Pose.create_from_pq(
            torch.tensor([[0.01, 0.0, -0.02]] * B, dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * B, dtype=torch.float32),
        )
        return overlapping * relative_pose

    def move_from_to_pose(
        self,
        robot_worldF: Pose,
        curr_joint_positions: torch.Tensor,
        from_pose: Pose,
        to_pose: Pose,
        closing: torch.Tensor,
        interpolate_steps: int = 0,
    ) -> list[torch.Tensor]:
        """1) Seed robot at `from_joints` 2) Read out current end-effector pose
        ("from_pose") 3) Build a Cartesian trajectory of length (interpolate_steps+1)
        *excluding* the start, i.e. fractions = [1/(n+1), …, 1] 4) Solve IK at each
        fraction to get a joint waypoint 5) Return list of joint arrays."""
        # ——— 1) seed & get “from” pose ———
        from_pos = from_pose.p.clone()  # (x,y,z)
        from_q = from_pose.q.clone()  # (w,x,y,z)

        # unpack goal
        to_pos = to_pose.p.clone()
        to_q = to_pose.q.clone()

        # ——— 2) if no interpolation, just solve final IK ———
        arm_joints = curr_joint_positions[:, :-1]
        gripper_pos = (curr_joint_positions[:, -1].unsqueeze(1)).clone()  # (B,1)
        gripper_pos[closing] += self.gripper_closing_delta
        gripper_pos[~closing] += self.gripper_openning_delta
        if interpolate_steps <= 0:
            sol = self.solve_spot_ik(
                base_pose=robot_worldF,
                end_effector_pose=to_pose,
                curr_joint_positions=arm_joints,
            )
            if sol is None:
                raise RuntimeError(f"IK failed for pose {to_pos}, {to_q}")
            # append gripper state
            sol = torch.cat([sol, gripper_pos], dim=1)
            return [sol]

        # ——— 3) build fractions [1/(n+1), …, 1] ———
        n = interpolate_steps
        fractions = [i / (n + 1) for i in range(1, n + 2)]

        traj: list[torch.Tensor] = []
        prev_joints = arm_joints.clone()
        a_vec = torch.zeros_like(from_q[:, 0])  # (B,)
        da = 1 / (n + 1)  # step size for a
        for i, a in enumerate(fractions):
            # Cartesian interp
            p = (1 - a) * from_pos + a * to_pos
            a_vec += da
            q = slerp_torch(from_q, to_q, a_vec)

            # ——— 4) solve IK at fraction ———
            to_pose = Pose.create_from_pq(p, q)

            sol = self.solve_spot_ik(
                base_pose=robot_worldF,
                end_effector_pose=to_pose,
                curr_joint_positions=prev_joints,
            )

            if sol is None:
                raise RuntimeError(f"IK failed at a={a:.2f} → pos={p}, quat={q}")

            # append gripper state
            prev_joints = sol.clone()
            sol = torch.cat([sol, gripper_pos], dim=1)
            traj.append(sol)

        return traj

    def open_gripper(self, curr_qpos: torch.Tensor, t: int = 6) -> list[torch.Tensor]:
        """Open the gripper from its current position over `t` steps."""
        qpos = curr_qpos[:, :-1]
        gripper_state = curr_qpos[:, -1].unsqueeze(1).clone()
        local_delta = (self.gripper_open - self.gripper_closed) / t
        actions = []
        for i in range(1, t + 1):
            curr_gripper_state = gripper_state + local_delta * i
            action = torch.cat([qpos, curr_gripper_state], dim=1)
            actions.append(action)
        return actions

    def close_gripper(self, curr_qpos: torch.Tensor, t: int = 6) -> list[torch.Tensor]:
        """Close the gripper from its current position over `t` steps."""
        qpos = curr_qpos[:, :-1]
        gripper_state = curr_qpos[:, -1].unsqueeze(1).clone()
        local_delta = (self.gripper_closed - self.gripper_open) / t
        actions = []
        for i in range(1, t + 1):
            curr_gripper_state = gripper_state + local_delta * i
            action = torch.cat([qpos, curr_gripper_state], dim=1)
            actions.append(action)
        return actions
