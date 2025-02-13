from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_aging': {
		'transforms': transforms_config.AgingTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['ffhq_val'],
		'test_target_root': dataset_paths['ffhq_val'],
	}
}

#
# DATASETS = {
# 	'morph_dataset': {
# 		'transforms': transforms_config.AgingTransforms,
# 		'train_source_root': dataset_paths['morph'],
# 		'train_target_root': dataset_paths['morph'],
# 		'test_source_root': dataset_paths['morph_val'],
# 		'test_target_root': dataset_paths['morph_val'],
# 	}
# }
