from typing import Optional
from ptlflow.data.flow_datamodule import FlowDataModule

class FlowDataModuleFlowBench(FlowDataModule):
    
    def __init__(self, *args, 
        kitti_2015_3DCC_root_dir: Optional[str] = None,
        kitti_2012_3DCC_root_dir: Optional[str] = None,
        mpi_sintel_3DCC_root_dir: Optional[str] = None,
        **kwargs, 
    ):
        super().__init__(*args, **kwargs)
        self.kitti_2015_3DCC_root_dir = kitti_2015_3DCC_root_dir
        self.kitti_2012_3DCC_root_dir = kitti_2012_3DCC_root_dir
        self.mpi_sintel_3DCC_root_dir = mpi_sintel_3DCC_root_dir
    