from pathlib import Path
import torch
from config import create_parser
from data_pipeline import unavailable
from loss_functions import DSECLLoss
from model import build_model

def build_training_interfaces(args):
    model = build_model(args.model_name, args.in_shape, Tout=args.out_T, hid_S=args.hid_S, N_S=args.N_S, N_T=args.N_T, weak_echo_thr=args.weak_echo_thr, adaptive_mask_prob=args.adaptive_mask_prob, daqo_time_kernel=args.temporal_attention_kernel, strong_echo_thr=args.strong_echo_thr, refine_mid_channels=args.refine_mid_channels, refiner_delta_scale=args.refiner_delta_scale, native_fill_teacher_forcing=args.teacher_forcing_end)
    criterion = DSECLLoss(alpha_early=args.alpha_early, beta_final=args.beta_final, lambda_mse=args.lambda_mse, lambda_ssim=args.lambda_ssim, lambda_edge=args.lambda_edge, lambda_strong=args.lambda_strong, lambda_temporal=args.lambda_temporal, strong_echo_thr=args.strong_echo_thr, strong_echo_cap=args.strong_echo_cap)
    return (model, criterion)

def train(model, criterion, train_loader, args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = min(1.0, max(0.0, (epoch - 1) / max(1, args.teacher_forcing_decay_epochs)))
        teacher_forcing = args.teacher_forcing_start + (args.teacher_forcing_end - args.teacher_forcing_start) * progress
        loss_sum = 0.0
        batch_count = 0
        for (step, (inputs, targets)) in enumerate(train_loader, start=1):
            if args.max_train_batches > 0 and step > args.max_train_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs, y_seq=targets, is_train=True, teacher_forcing_ratio=teacher_forcing)
            (loss, _) = criterion(outputs['coarse'], outputs['final'], targets)
            (loss / args.accum_steps).backward()
            if step % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            batch_count += 1
        if batch_count % args.accum_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'loss': loss_sum / max(1, batch_count)}
        torch.save(checkpoint, checkpoint_dir / 'last.pth')
        print(f"Epoch {epoch}/{args.epochs} loss={checkpoint['loss']:.6f} lr={scheduler.get_last_lr()[0]:.8f} teacher_forcing={teacher_forcing:.4f}")
    return model

def main():
    args = create_parser().parse_args()
    build_training_interfaces(args)
    unavailable()
if __name__ == '__main__':
    main()
