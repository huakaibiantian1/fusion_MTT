"""检查模型权重统计"""
import torch

# 加载检查点
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu', weights_only=False)

print(f'训练步数: {ckpt["step"]}')
print(f'验证OSPA: {ckpt.get("val_ospa", "N/A")}')

# 检查输出层权重
model_dict = ckpt['model_state_dict']
output_weight = model_dict['state_output_head.weight']
output_bias = model_dict['state_output_head.bias']

print(f'\nstate_output_head.weight 统计:')
print(f'  均值: {output_weight.mean():.6f}')
print(f'  标准差: {output_weight.std():.6f}')
print(f'  范围: [{output_weight.min():.6f}, {output_weight.max():.6f}]')
print(f'  形状: {output_weight.shape}')

print(f'\nstate_output_head.bias 统计:')
print(f'  均值: {output_bias.mean():.6f}')
print(f'  标准差: {output_bias.std():.6f}')
print(f'  范围: [{output_bias.min():.6f}, {output_bias.max():.6f}]')

# 检查embedding层权重
state_emb_weight = model_dict['state_embedding.weight']
meas_emb_weight = model_dict['measurement_embedding.weight']

print(f'\nstate_embedding.weight 统计:')
print(f'  均值: {state_emb_weight.mean():.6f}')
print(f'  标准差: {state_emb_weight.std():.6f}')
print(f'  范围: [{state_emb_weight.min():.6f}, {state_emb_weight.max():.6f}]')

print(f'\nmeasurement_embedding.weight 统计:')
print(f'  均值: {meas_emb_weight.mean():.6f}')
print(f'  标准差: {meas_emb_weight.std():.6f}')
print(f'  范围: [{meas_emb_weight.min():.6f}, {meas_emb_weight.max():.6f}]')
