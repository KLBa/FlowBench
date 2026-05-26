from typing import Optional
from ptlflow.data.flow_datamodule import FlowDataModule
from .datasets_3DCC import (
    KittiDataset3DCC,
    SintelDataset3DCC,
)

class PrecomputedDataset:
    
    def __init__(self, dataset, dataset_perturbed):
        super().__init__()
        self.dataset = dataset
        self.dataset_perturbed = dataset_perturbed
        
        assert len(self.dataset) == len(self.dataset_perturbed), f"{len(self.dataset), len(self.dataset_perturbed)} The unperturbed dataset must match the perturbed dataset but they're differently sized."

    def __getitem__(self, index: int):
        # We zip the two datasets, i.e.,
        # just assume same index = same frame pair
        inputs = self.dataset[index]
        inputs_perturbed = self.dataset_perturbed[index]
        inputs['inputs_perturbed'] = inputs_perturbed
        
        assert inputs["images"].device == inputs_perturbed["images"].device
        return inputs

    def __len__(self) -> int:
        return len(self.dataset)


class FlowDataModuleFlowBench(FlowDataModule):
    
    def __init__(self, *args, 
        kitti_2012_3DCC_root_dir: Optional[str] = None,
        kitti_2015_3DCC_root_dir: Optional[str] = None,
        mpi_sintel_3DCC_root_dir: Optional[str] = None,
        **kwargs, 
    ):
        super().__init__(*args, **kwargs)
        self.kitti_2012_3DCC_root_dir = kitti_2012_3DCC_root_dir
        self.kitti_2015_3DCC_root_dir = kitti_2015_3DCC_root_dir
        self.mpi_sintel_3DCC_root_dir = mpi_sintel_3DCC_root_dir

    def _get_kitti_precomputed_dataset(self, is_train: bool, *args: str):
        args_tdcc = [arg for arg in args if arg.startswith('tdcc') or arg.startswith('flowbench_')]
        args_dataset = [arg for arg in args if arg not in args_tdcc]
        dataset_kitti = super()._get_kitti_dataset(is_train, *args_dataset)

        is_tdcc = False
        attach = False
        for v_ in args_tdcc:
            v = v_.replace('flowbench_', '')
            if 'tdcc_' in v:
                is_tdcc = True
                v = v.replace('tdcc_', '')
                tdcc_intensity = v[0]
                assert v[1] == '_'
                tdcc_corruption = v[2:]
            elif v == 'attach':
                attach = True
            else:
                raise ValueError(f"Invalid arg: {v_}")

        if is_tdcc:
            dataset_tdcc = KittiDataset3DCC(
                self.kitti_2012_root_dir,
                self.kitti_2015_root_dir,
                self.kitti_2012_3DCC_root_dir,
                self.kitti_2015_3DCC_root_dir,
                tdcc_corruption=tdcc_corruption,
                tdcc_intensity=tdcc_intensity,
                versions=dataset_kitti.versions,
                split=dataset_kitti.split,
                transform=dataset_kitti.transform,
            )
            dataset = dataset_tdcc

            if attach:
                dataset = PrecomputedDataset(dataset_kitti, dataset_tdcc)
        else:
            dataset = dataset_kitti
        return dataset

    def _get_sintel_precomputed_dataset(self, is_train: bool, *args: str):
        args_tdcc = [arg for arg in args if arg.startswith('tdcc') or arg.startswith('flowbench_')]
        args_dataset = [arg for arg in args if arg not in args_tdcc]
        dataset_sintel = super()._get_sintel_dataset(is_train, *args_dataset)

        is_tdcc = False
        attach = False
        for v in args_tdcc:
            v = v.replace('flowbench_', '')
            if 'tdcc_' in v:
                is_tdcc = True
                v = v.replace('tdcc_', '')
                tdcc_intensity = v[0]
                assert v[1] == '_'
                tdcc_corruption = v[2:]
            if v == 'attach':
                attach = True
            else:
                raise ValueError(f"Invalid arg: {v}")

        if is_tdcc:
            dataset_tdcc = SintelDataset3DCC(
                self.mpi_sintel_root_dir,
                self.mpi_sintel_3DCC_root_dir,
                tdcc_corruption=tdcc_corruption,
                tdcc_intensity=tdcc_intensity,
                split=dataset_sintel.split,
                pass_names=dataset_sintel.pass_names,
                get_occlusion_mask=dataset_sintel.get_occlusion_mask,
                sequence_length=dataset_sintel.sequence_length,
                sequence_position=dataset_sintel.sequence_position,
                transform=dataset_sintel.transform,
            )
            dataset = dataset_tdcc

            if attach:
                dataset = PrecomputedDataset(dataset_sintel, dataset_tdcc)
        else:
            dataset = dataset_sintel
        return dataset
