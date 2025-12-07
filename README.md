# AquaEdge
这是我们提出的AquaEdge项目代码。/This is the AquaEdge project code we have proposed.
Abstract—Acquisition of real-time river surface flow velocity and water body extent is crucial for flood control, water resource management, and ecological protection. However,
existing technologies often rely on high-performance servers,
making “on-site, near real-time” analysis challenging, especially in communication-constrained emergency scenarios. This
limitation primarily stems from current approaches treating
semantic segmentation and optical flow estimation as independent tasks, relying on ground workstations for post-processing,
which prevents low-latency, integrated inference on the drone
edge. To address this, this paper proposes AquaEdge, an endto-end drone edge intelligence system integrating lightweight
semantic segmentation and optical flow estimation. The system
first introduces a segmentation network that combines a shared
backbone with a Frequency-augmented Star Block (FAStar) and
a lightweight module named Cross Stage Partial with Partial
Multi-Dilated Depthwise Convolution and Channel Attention
(CSPPC-MDCA) to achieve efficient fusion of global and local
features. Second, it designs the NueFlow-lite lightweight optical
flow inference algorithm, specifically optimized for accurate
surface flow velocity estimation on edge devices. Furthermore,
to fill the data gap in this domain, we constructed and opensourced the Heihe River Basin comprehensive channel image
dataset, providing a solid foundation for model training and
evaluation. Experimental results show that AquaEdge achieves a
97% water body segmentation accuracy, an average relative error
of 13% for surface flow velocity estimation, and an end-to-end
inference latency of approximately 200 ms on the drone platform.
This demonstrates the system’s capability to competently handle
quasi-real-time, integrated river channel hydrological sensing
tasks. This study offers a reliable, rapid assessment method for
both routine monitoring and disaster emergency response.
