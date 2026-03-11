# Advanced Objective Functions for Monocular Elevation Regression: Optimizing ControlNet Architectures for nDSM Generation

## Introduction to the Challenges of Monocular Elevation Regression

The generation of a normalized Digital Surface Model (nDSM) from a standard Digital Surface Model (DSM) using monocular imagery represents a highly complex, fundamentally ill-posed problem in modern photogrammetry, remote sensing, and computer vision.1 An nDSM is theoretically derived by subtracting a Digital Terrain Model (DTM)—which represents the bare earth—from a DSM, thereby isolating the absolute heights of above-ground features such as buildings, dense vegetation, and urban infrastructure.4 While traditional derivations rely on dense Light Detection and Ranging (LiDAR) point clouds, multi-view stereopsis, or synthetic aperture radar (SAR) interferometry, these methods are consistently constrained by prohibitive acquisition costs, limited temporal resolution, and highly complex data pairing requirements.1 Structure-from-Motion (SfM) techniques similarly demand sophisticated acquisition techniques and rely on precise triangulations from consecutive overlapping views, rendering them unsuitable for instantaneous, single-image inference.1 Consequently, leveraging deep neural networks to infer nDSM data directly from a single DSM or an optical image has emerged as a compelling, highly scalable, and cost-effective alternative.1

The recent deployment of latent diffusion models, specifically Stable Diffusion 1.5 augmented by ControlNet architectures, has completely revolutionized the landscape of conditional image synthesis and spatial translation tasks.9 ControlNet operates by freezing the pre-trained weights of the generative diffusion model while learning task-specific conditions through a trainable replica of the encoding layers, which are bridged by zero-convolutions.12 This sophisticated architectural mechanism prevents the injection of random noise gradients during the early training phases, effectively protecting the robust, large-scale semantic priors learned by the diffusion backbone across billions of images.13 Furthermore, fine-tuning the Variational Autoencoder (VAE) to process one-channel spatial data enables the diffusion model to output deterministic, continuous elevation values rather than stochastically generated multi-channel RGB textures.10

Despite these profound architectural advancements, transitioning a generative text-to-image model to perform strict, pixel-perfect spatial regression introduces a myriad of profound optimization bottlenecks. The current approach of utilizing pixel-level losses and basic gradient losses improves structural fidelity over pure latent-based losses, yet it consistently encounters two critical, interrelated failure modes in practice. First, the model inherently struggles with the extreme height discontinuities and sharp topographical transitions characteristic of urban and forested environments.3 Unlike natural photographic images where color and lighting gradients transition smoothly and continuously, elevation maps contain sharp, Heaviside-like step functions at the boundaries of distinct objects.18 Standard convolutional operators within the U-Net naturally aggregate features across these spatial discontinuities, resulting in smoothed, blurred transitions rather than sharp architectural edges.16 The model fails to comprehend the geometric reality of the scene—specifically, which side of a detected edge belongs to the elevated object and which side belongs to the terrain—leading to severe morphological bleeding and structurally unsound height predictions.

Second, the prediction of nDSMs suffers from an extreme class-imbalance problem situated within the continuous regression domain.20 In virtually any given aerial or satellite scene, the ground (where the nDSM value is exactly or near zero) overwhelmingly dominates the spatial footprint, often comprising between 80% to 95% of the total image pixels. Because standard loss functions compute the arithmetic mean of errors across the entire tensor, the optimization landscape becomes heavily, almost entirely, biased toward accurately predicting the ground.23 Consequently, global evaluation metrics such as Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and Mean Absolute Error (MAE) report deceptively low values, suggesting high model performance.3 However, human visual inspection and rigorous downstream geometric analyses reveal that the minority features—the actual objects of interest representing the varying height differences—are severely degraded, structurally collapsed, or completely hallucinated.

Resolving these dual challenges requires a paradigm shift, moving far beyond standard isotropic loss functions and naive pixel-wise comparisons. The optimization objective of the ControlNet must be mathematically restructured to explicitly penalize blurred structural boundaries, definitively encode the orientation and direction of spatial gradients, and dynamically reweight the loss landscape to counter the immense statistical dominance of the ground plane.

## The Pathology of Global Error Metrics in Spatial Regression

To fully comprehend why a ControlNet model can achieve exceptionally low MSE and MAE while simultaneously producing visually inadequate nDSM predictions, it is necessary to rigorously examine the statistical mechanics of regression over highly skewed spatial distributions. In standard machine learning paradigms, regression tasks implicitly assume that the target variables are uniformly or normally distributed across the continuous spectrum.21 In the highly specialized context of nDSM generation, the target distribution is characterized by a massive, overwhelming delta spike at zero meters (representing the ground plane) and a sparse, highly fragmented, long-tailed distribution representing varying object heights.4

When a model is supervised by an unweighted MSE or MAE loss, the parameter updates during backpropagation are dictated by the global gradient of the error surface.27 Because the vast majority of pixels belong to the ground class, the cumulative gradient magnitude generated by minor, almost imperceptible fluctuations in ground prediction mathematically dwarfs the gradient magnitude generated by significant, catastrophic errors in building or tree height prediction.23 The neural network functions as an aggressive, highly efficient loss-minimization engine; it rapidly learns that predicting a uniform flat surface, or an overly smoothed, conservative approximation of the terrain, is the mathematically safest strategy to minimize the global loss function.24

This phenomenon fundamentally explains the paradox of the current model's behavior: the numerical loss is small, yet the visual output is poor. Standard metrics like RMSE and MAE treat every spatial pixel equally, entirely failing to capture perceptually and geometrically relevant differences, particularly at high-frequency structural details like rooflines, building footprints, and tree canopies.3 When the ground plane dictates the entire trajectory of gradient descent, the model experiences what is known as "Simulation Correlation," a phenomenon closely akin to long-tailed classification problems where the network effectively ignores the minority domain entirely to satisfy the majority gradient.33 To force the ControlNet to render sharp, accurate object heights and respect the geometry of the scene, the loss function must be completely decoupled from the raw spatial pixel frequency.

Furthermore, traditional image-based global statistical metrics provide limited insight into depth estimation quality in areas with depth discontinuities.3 This limitation consistently results in similar error scores for models that differ significantly in their actual ability to capture complex spatial structures in built or forest environments.3 Metrics such as the Structural Similarity Index Measure (SSIM) often saturate when dealing with raw elevation data, providing almost perfect values for poorly reconstructed DEMs due to the massive dynamic range of the background.25 Therefore, relying on these traditional metrics to gauge the performance of a ControlNet generating an nDSM is highly deceptive, necessitating a transition toward geometry-aware metrics and cost-sensitive loss formulations.

## Addressing Class Imbalance in Continuous Regression

The problem of an overwhelmingly dominant ground class cannot be solved by simply providing the model with more training data, as the inherent distribution of the physical world guarantees that ground pixels will always outnumber object pixels. Instead, the loss function itself must be altered to implement cost-sensitive learning or targeted reweighting.

### Implementing Balanced Mean Squared Error (BMSE)

The most robust theoretical solution to the problem of continuous label imbalance in vision tasks is the Balanced Mean Squared Error (BMSE).21 BMSE fundamentally reframes imbalanced regression from a purely numerical error calculation into a statistical generalization problem.21 Traditional MSE forces the regressor to model the training label distribution directly. If the training data is composed of 90% ground pixels, the network's internal priors become heavily biased toward predicting the ground.22 BMSE leverages the training label distribution prior to execute a sophisticated statistical conversion, allowing the model to minimize the negative log-likelihood of a balanced test distribution while still training on the heavily imbalanced data.22

In practice, the implementation of BMSE for nDSM generation requires discretizing the continuous height space into a predefined number of bins (e.g., 0m, 1m-5m, 5m-10m, etc.) to empirically estimate the probability density function of the elevation values.36 Once the empirical frequency of each height bin is calculated over the dataset, BMSE modifies the standard loss calculation by weighting the squared error of a prediction inversely proportional to the frequency of its ground-truth bin.24 Continuous labels can be boundless and high-dimensional, making them inherently more challenging to balance than discrete classification categories; BMSE is specifically designed to be the first general solution to this high-dimensional imbalanced regression context.21

However, unlike simple inverse frequency weighting—which can easily destabilize a network if a rare sample produces an infinite weight—BMSE incorporates a theoretically sound margin that prevents the gradients of extremely rare height values from exploding and destabilizing the VAE decoder or the diffusion model's U-Net.21 By applying BMSE, the ControlNet is mathematically compelled to treat a 1-meter error on a 50-meter building with the exact same urgency and gradient magnitude as a 1-meter error on the ground, drastically improving the rendering of the elevated structures without artificially altering the dataset via undersampling.21

### Focal Regression and Density-Based Relevance

An alternative, highly effective, or potentially complementary approach to BMSE is Focal Regression Loss. This is an adaptation of the highly successful Focal Loss initially designed for dense object detection to address extreme foreground-background imbalances, scaled for continuous domains.40 Traditional Focal Loss dynamically scales the cross-entropy penalty based on prediction confidence, heavily down-weighting the loss for easily classified examples (in this context, the flat ground) and aggressively focusing the training on hard, misclassified examples (the object boundaries and heights).30

To adapt this specifically for continuous elevation regression, Focal Regression modifies the standard L1 or L2 norm by incorporating a dynamic difficulty term into the polynomial.40 If the predicted height is far from the true height, indicating a high absolute error, the penalty is amplified by an exponent. Conversely, if the predicted height is close to the true height—meaning the model has easily predicted the flat ground—the loss is exponentially decayed toward zero.41 This automated adjustment ensures that the bulk of the backpropagated gradients originate from the poorly reconstructed objects rather than the vast, easily predicted ground.41 Automated Focal Loss formulations even design dynamically adjusted beta parameters to fit the model convergence during the training process, preventing decreasing gradients as the model learns.41

Furthermore, Density-Based Relevance (DBR) can be utilized to map the specific rarity of voxel contents or pixel heights directly to a cost-sensitive learning framework.26 Using Kernel Density Estimation (KDE) on the nDSM dataset, a continuous probability density function of the terrain heights is established prior to training.44 The relevance weight assigned to each pixel during the loss calculation is inversely proportional to its density value.44 To avoid the extremes of pure inverse weighting, which can cause erratic training behavior, the weights are typically normalized and squarerooted to compress the dynamic range, providing a smoothed, highly targeted penalty that strictly prioritizes the minority object classes without causing numerical overflow.44


## Geometric Precision: Defining Object versus Ground Boundaries

Solving the class imbalance problem merely guarantees that the model will actively attempt to predict objects, but it does not inherently guarantee that those objects will possess correct, sharp geometric boundaries. The observation that "height changes dramatically due to height differences" touches upon the fundamental architectural vulnerability of convolutional neural networks: they inherently struggle with high-frequency spatial discontinuities. In an nDSM, the boundary of a building represents a near-vertical drop-off—a mathematical singularity where the gradient approaches infinity.16

Standard pixel-level loss (MSE) and standard gradient loss (measuring the raw magnitude of Sobel or Prewitt filter responses) are entirely insufficient for differentiating which side of the edge belongs to the object and which side belongs to the ground. A standard gradient magnitude loss simply dictates that an edge exists somewhere in the vicinity; it is completely isotropic, meaning it penalizes the absence of an edge but provides no directional vector indicating whether the height is stepping up or stepping down.45 Consequently, the diffusion model may generate a building with the correct general height but with blurred, sloping walls that bleed into the ground, or it may displace the height drop-off entirely, causing the object footprint to expand.

### Direction-Aligned and Sign-Preserving Gradient Loss

To explicitly detect which pixel constitutes the edge and definitively establish the interior (the object) versus the exterior (the ground), the objective function must transition from an isotropic gradient magnitude penalty to a Direction-Aligned, Sign-Preserving Gradient Loss.45

Spatial gradients inherently encode local structures and transitions. When computing the spatial derivative of an elevation map along the X and Y axes, the resulting vector contains both a magnitude (the steepness of the terrain) and a sign (the direction of the step).45 If a building is taller than the ground, moving from the ground into the building yields a massive positive gradient, while moving from the building to the ground yields a massive negative gradient.46

Standard gradient losses maliciously collapse this crucial vector into a scalar magnitude using the Euclidean norm (), obliterating the directional sign and rendering the supervision ambiguous.46 By contrast, a Direction-Aligned Gradient Loss computes the horizontal () and vertical () derivatives completely separately and penalizes the discrepancy between the predicted and ground-truth derivatives while strictly preserving their sign.45 This forces the neural network to intimately understand the spatial orientation of the boundary.

If the ControlNet attempts to slope the building into the ground (creating a wide band of small positive gradients) rather than creating a sharp, single-pixel cliff (a single pixel of massive positive gradient), the sign-preserving loss heavily penalizes the spatial spread of the derivative.46 This mechanism directly and elegantly answers the query regarding how to detect which side of the edge is the object. The positive and negative polarity of the decoupled  and  tensors acts as an absolute geometric compass, actively constraining the diffusion model's U-Net to align the sharp height transition precisely with the spatial coordinates of the object footprint.46

### Laplacian Regularization and Second-Order Derivatives

While first-order derivatives (gradients) describe the slope of the elevation, second-order derivatives, calculated via the Laplacian operator (), describe the rate of change of the slope. Laplacian loss is paramount for modeling height map discontinuities because it is exquisitely sensitive to the exact inflection point of a structural boundary.17

In flat regions, whether on the ground or atop a flat roof, the Laplacian evaluates to exactly zero. At the exact pixel where the building drops to the ground, the Laplacian exhibits a massive positive spike immediately adjacent to a massive negative spike—a geometric zero-crossing. By integrating a Laplacian Loss term, mathematically defined as , the model is penalized heavily if the inflection point of the predicted edge does not perfectly and exactly overlap with the ground truth.47

The  norm is specifically preferred over the  norm for the Laplacian penalty because  naturally promotes sparsity; it encourages the network to consolidate the edge into a single, razor-sharp pixel boundary rather than a smeared transition zone.48 The use of Laplacian regularization directly pulls each vertex toward the center of its neighbors unless a true discontinuity exists, enforcing smooth 3D surfaces on the roof and the ground while allowing for infinite slope at the walls.47

### Surface Normal and Cross-Product Consistency

To ensure that the vertical walls of the objects are perfectly orthogonal to the ground plane, thereby separating the height difference explicitly between object and ground, Surface Normal Loss must be integrated into the regression framework.47 The surface normal vector at any pixel in the nDSM can be derived geometrically through the cross-product of the partial derivatives. By normalizing the cross product of the gradients ( and ), a 3D vector representing the absolute perpendicular orientation of the surface at that specific pixel is obtained.51

When minimizing the cosine distance between the predicted surface normals and the ground-truth surface normals, the model is forcibly constrained to respect the 3D geometry of the physical scene.50 The ground will possess normals pointing strictly upward along the Z-axis. The edges of buildings will possess normals pointing strictly horizontally along the X or Y axes. If the model attempts to generate a blurred, sloping edge to minimize standard MSE, the resulting normal vectors will point at an anomalous 45-degree angle, triggering a massive cosine penalty.51

This explicit geometric constraint separates the height differences by enforcing piecewise planar structures, ensuring that roofs remain entirely flat, walls remain strictly vertical, and the ground remains perfectly horizontal.50 The depth-normal consistency loss effectively achieves highly accurate depth estimation by using the normal as a strict guide to capture neighboring pixel-visibility.56


## Implicit Spatial Representations for Boundary Disambiguation

Beyond manipulating the raw gradients of the pixel output, advanced techniques originating from 3D reconstruction and computer graphics can be adapted to monocular elevation regression to explicitly model and enforce boundary delineations. Two prominent, highly effective methodologies are Signed Distance Fields (SDF) and Displacement Fields.

### Signed Distance Fields (SDF) for Explicit Boundary Supervision

A Signed Distance Field (SDF) is a continuous implicit function where the value of each spatial coordinate represents its exact orthogonal distance to the nearest boundary.58 Crucially, the sign of the value strictly dictates whether the coordinate is inside the object (e.g., negative values) or outside the object (e.g., positive values), and the exact physical boundary exists perfectly at the zero-crossing.59

In the specific context of nDSM prediction, an SDF can be pre-calculated from the ground-truth elevation map before training begins. Every pixel belonging to a building or object is assigned a negative distance value relative to how deep it is inside the building footprint. Conversely, every ground pixel is assigned a positive value relative to how far it is from the nearest building.59

By adding an auxiliary prediction head to the ControlNet decoder that predicts the 2D SDF map alongside the 1-channel nDSM height map, the network is forced to learn a highly structured, object-centric latent representation.58 The SDF loss acts as an extraordinarily powerful topological constraint. Because the SDF explicitly encodes the interior versus the exterior strictly through its sign, and the distance to the edge through its magnitude, it perfectly and elegantly addresses the user's need to "detect which pixel is the edge, or which side of the edge is the object, which side is the ground".60 Furthermore, predicting the SDF encourages the diffusion model to understand the scene not as an arbitrary array of independent pixels, but as a collection of discrete, coherent geometric entities resting upon a continuous terrain.62

### Displacement Fields for Occlusion Boundary Refinement

An alternative to predicting implicit distance functions is the use of learned Displacement Fields for post-hoc boundary sharpening.64 Current monocular depth estimation methods frequently predict smooth, poorly localized contours around occlusion boundaries due to the inherent receptive field expansion of deep convolutional networks.64 To rectify this without relying on complex, non-differentiable post-processing like Conditional Random Fields (CRFs), a displacement field module can be appended to the network architecture.64

Given a slightly blurred initial nDSM prediction from the ControlNet, a secondary lightweight convolutional block learns a 2D displacement field. This field contains sub-pixel vector coordinates instructing the network on exactly how to re-sample pixels around the occlusion boundaries to synthesize a demonstrably sharper reconstruction.64 Essentially, if the network detects a blurred height transition, the displacement field forcefully instructs the pixels on the slope to snap either to the higher elevation (the object) or the lower elevation (the ground), artificially steepening the gradient to near infinity.64 Because this entire process relies on bilinear interpolation, it remains fully differentiable, allowing for end-to-end training while drastically sharpening the separation between object and ground.64

## Architectural Implications for ControlNet and Stable Diffusion 1.5

Integrating these highly complex geometric, statistical, and implicit loss functions into a Stable Diffusion 1.5 backbone controlled via a ControlNet adapter requires specific architectural handling. This is particularly crucial given the reliance on a 1-channel Variational Autoencoder (VAE) for continuous elevation regression.

### VAE Latent Space Dynamics and Bit-Depth Quantization

Stable Diffusion 1.5 natively operates in a highly compressed latent space facilitated by an 8-bit, 3-channel (RGB) VAE.13 When fine-tuning this VAE to process a 1-channel spatial map—where pixel intensity corresponds directly to absolute height rather than visual color—researchers must carefully, rigorously manage precision and quantization dynamics.10

Standard 8-bit integer precision allows for only 256 discrete values. If an nDSM represents a terrain ranging from 0 meters to 100 meters, an 8-bit depth map forces a quantization step of approximately 0.4 meters.66 This low-bit precision natively induces severe "staircasing" artifacts on pitched roofs, slopes, or subtle terrain variations. These quantization artifacts introduce false edges that will severely disrupt and corrupt the Gradient, Laplacian, and Surface Normal losses discussed previously, as the loss functions will penalize the network for failing to match artifactual steps.65

To preserve the continuous fidelity of the elevation data and prevent the loss functions from penalizing artifactual quantization boundaries, the 1-channel VAE must absolutely be trained and inferred using 16-bit floating-point (FP16) or ideally 32-bit floating-point (FP32) tensors.65 Running the VAE encoding/decoding process in higher bit-depth ensures that the network possesses the dynamic range necessary to express perfectly smooth micro-gradients on flat surfaces while retaining the massive capacity to express infinite discontinuities at object boundaries.65

In highly constrained scenarios where hardware limitations strictly force low-precision quantization on edge deployments, recent pioneering research demonstrates that high dynamic range depth can be mapped onto a 2D Hilbert space-filling curve prior to processing.66 By predicting the spatial coordinates of a continuous Hilbert curve rather than raw scalar height, the model effectively preserves profound spatial correlation and limits the catastrophic destruction of edge discontinuities during bit-width compression.67 While mathematically complex, embedding Hilbert curve parameters provides a robust mechanism to maintain 32-bit effective precision within a highly optimized, low-bit pipeline.66

### Boundary-Aware Perceptual Loss in the Latent Domain

A defining, critical feature of latent diffusion models is that the actual denoising U-Net operates entirely in the latent space, while pixel-level losses (such as MSE, MAE, and Directional Gradients) must be computed in the decoded pixel space.13 Relying solely on pixel-level loss backpropagated through a frozen or finetuned VAE decoder can frequently lead to suboptimal feature matching, as the VAE acts as an informational bottleneck that can dilute high-frequency gradients before they reach the U-Net.15

To bridge this latent-pixel gap, an Edge-Guided or Boundary-Aware Perceptual Loss should be heavily utilized.70 Traditional perceptual loss extracts feature maps from a pre-trained network (such as VGG-16) and compares the intermediate layer activations of the predicted image against the ground truth, optimizing for overall semantic layout and structure rather than raw pixel differences.31

However, for nDSM generation, standard VGG features (which are trained on natural RGB images) are demonstrably less effective, as they look for textures and colors rather than elevation geometries.74 Instead, an auxiliary Boundary-Aware Perceptual Loss utilizes a custom, lightweight loss network (e.g., CycleNet) trained specifically to transfer structural edge embeddings.70 By enforcing feature-space similarity exactly at the boundary locations, the perceptual loss injects high-level structural context directly into the latent space of the ControlNet.70 It teaches the diffusion model to recognize the gestalt of an object—such as the perfectly rectangular footprint of a building or the circular canopy of a tree—preventing the generation of fractured, incomplete, or morphologically implausible height artifacts.71

### Balancing Zero-Convolutions and Loss Trajectories

The core architectural efficiency of ControlNet relies on "zero-convolutions"— convolutional layers with weights and biases explicitly initialized to zero—that progressively introduce the conditioning spatial data to the frozen diffusion backbone.11 At the exact beginning of training, the ControlNet branch outputs zero, meaning the loss relies purely on the diffusion model's pre-trained prior.11

When applying highly aggressive, mathematically rigid geometric losses like the Laplacian or Surface Normal loss, the gradients during the first few epochs can be violently erratic, potentially destabilizing the zero-convolution initialization entirely.11 To ensure stable convergence, a strict curriculum learning strategy must be employed. The training should commence exclusively with the Balanced MSE (to establish general height and combat ground dominance) and a simple first-order Directional Gradient loss.21

Only after the zero-convolutions have sufficiently warmed up, and the network reliably approximates the general topography of the scene, should the higher-order geometric penalties (Laplacian, Surface Normals, and SDF constraints) be linearly annealed into the total objective function. This phased, careful approach guarantees that the model learns global coherence before being violently penalized for microscopic boundary imprecisions.47


## Designing the Comprehensive Multi-Objective Function

Based on the exhaustive theoretical analysis of imbalanced spatial regression, differential geometry, and implicit spatial representations, the optimal solution for predicting nDSMs via ControlNet requires synthesizing the aforementioned techniques into a unified, mathematically robust multi-objective loss function. The total loss  must concurrently manage pixel scale, structural orientation, edge sharpness, and geometric orthogonality, without allowing any single metric to collapse the others.

The recommended formulation is a weighted, dynamically adjusted summation of four distinct loss families:

The Imbalance Anchor (Balanced MSE): This entirely replaces the standard  pixel loss. By weighting the error based on the inverse frequency of the height bin (determined via KDE), it mathematically guarantees that ground pixels do not dominate the optimization trajectory. This ensures that the numerical loss accurately reflects the visual fidelity of the minority object classes.21

The Orientation Guide (Direction-Aligned Gradient Loss): This replaces the isotropic gradient magnitude loss. By independently assessing the  and  derivatives while strictly preserving their mathematical signs, it definitively teaches the network which side of the transition is the elevated object (positive gradient) and which is the terrain (negative gradient).45

The Sharpness Enforcer (Laplacian / Edge-Preserving Loss): This employs the  norm of the second spatial derivative () alongside Total Variation (TV) regularization to heavily penalize sloping, blurred walls. This forces the model to consolidate height drops into sharp, single-pixel step-function discontinuities.48

The Geometric Validator (Surface Normal / SDF Loss): This utilizes the cross-product of the spatial gradients to compute 3D surface normals, minimizing the cosine divergence against ground-truth normals. Alternatively, it enforces an SDF auxiliary prediction to topologically encode the interior boundaries of the objects.51

Mathematically, the entire objective can be expressed as:


The dynamic weighting coefficients () should not remain static. They must be adaptively modulated using a mechanism like homoscedastic task uncertainty or a linear step schedule.72 This explicitly prevents the heavily magnified gradient penalties from overwhelming the base height estimation during the initial, fragile phases of diffusion model training.11

## Conclusion

The extraction of accurate, sharp nDSMs from single monocular DSMs using ControlNet and Stable Diffusion 1.5 is fundamentally limited by the mathematical inadequacies of standard regression losses. When ground pixels comprise the vast majority of the spatial domain, unweighted loss functions like MSE and MAE produce visually degraded output that is entirely masked by deceptively low error metrics, resulting in a model that aggressively and incorrectly favors flat topographies.

By replacing naive pixel loss with Balanced Mean Squared Error (BMSE), Focal Regression, or Density-Based Relevance, the optimization landscape is statistically restructured, forcing the diffusion model to prioritize the accurate elevation synthesis of sparse objects such as buildings and vegetation. Furthermore, the persistent challenge of dramatic, blurred height transitions can be explicitly resolved by transitioning from scalar gradient losses to Direction-Aligned, Sign-Preserving Gradient losses. These decoupled derivatives, combined with Laplacian regularization and Surface Normal consistency, provide the necessary geometric vectors to inform the neural network exactly where an edge exists, which side constitutes the object, and how sharp the drop-off must physically be.

When properly integrated with a 1-channel, high-bit-depth VAE to prevent catastrophic quantization artifacts, and supervised by an auxiliary Signed Distance Field, this multi-objective framework guarantees that the latent generative capabilities of Stable Diffusion are rigidly constrained by the physical geometry of the terrain. The resulting architecture transcends the limitations of standard generative models, yielding nDSMs with unparalleled structural fidelity, perfectly orthogonal boundaries, and absolute spatial accuracy.

#### Nguồn trích dẫn

IMG2nDSM: Height Estimation from Single Airborne RGB Images, truy cập vào tháng 2 23, 2026, https://www.mdpi.com/2072-4292/13/12/2417

Deep Neural Network Regression for Normalized ... - IEEE Xplore, truy cập vào tháng 2 23, 2026, https://ieeexplore.ieee.org/iel7/4609443/9973430/10189905.pdf

A Comprehensive Evaluation of Monocular Depth Estimation ... - MDPI, truy cập vào tháng 2 23, 2026, https://www.mdpi.com/2072-4292/17/4/717

nDSMs: How digital surface models and digital terrain ... - UP42, truy cập vào tháng 2 23, 2026, https://up42.com/blog/ndsms-how-digital-surface-models-and-digital-terrain-models-elevate-your

GrounDiff: Diffusion-Based Ground Surface Generation from Digital, truy cập vào tháng 2 23, 2026, https://arxiv.org/html/2511.10391v1

(PDF) Digital Surface Model Super-Resolution by Integrating High, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/publication/380497790_Digital_Surface_Model_Super-Resolution_by_Integrating_High-Resolution_Remote_Sensing_Imagery_Using_Generative_Adversarial_Networks

DEEP LEARNING-BASED DIGITAL SURFACE MODEL (DSM, truy cập vào tháng 2 23, 2026, https://essay.utwente.nl/fileshare/file/97348/Abdela_MA_ITC.pdf

Building Extraction and Floor Area Estimation at the Village Level in, truy cập vào tháng 2 23, 2026, https://www.mdpi.com/2072-4292/14/20/5175

Train your ControlNet with diffusers - Hugging Face, truy cập vào tháng 2 23, 2026, https://huggingface.co/blog/train-your-controlnet

Altering ControlNet Input: A Guide for Input Adjustment - Bria Blog, truy cập vào tháng 2 23, 2026, https://blog.bria.ai/altering-controlnet-input-a-guide-for-input-adjustment

arXiv:2302.05543v1 [cs.CV] 10 Feb 2023 - deepsense.ai, truy cập vào tháng 2 23, 2026, https://deepsense.ai/wp-content/uploads/2023/04/2302.05543.pdf

LightDiffNet A diffusion-based predictive model for daylight, truy cập vào tháng 2 23, 2026, https://papers.cumincad.org/data/works/att/caadria2025_410.pdf

Low-Rank Conditioning for Diffusion-Based Post-Flood Satellite, truy cập vào tháng 2 23, 2026, https://cs231n.stanford.edu/2025/papers/text_file_840585726-CS231N_Final_Report.pdf

What are Diffusion Models? | Lil'Log, truy cập vào tháng 2 23, 2026, https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

Variational Autoencoder (VAE): How to train and inference (with code), truy cập vào tháng 2 23, 2026, https://www.reddit.com/r/StableDiffusion/comments/1ojhzgf/variational_autoencoder_vae_how_to_train_and/

DAR-MDE: Depth-Attention Refinement for Multi-Scale Monocular ..., truy cập vào tháng 2 23, 2026, https://www.mdpi.com/2224-2708/14/5/90

LEX v1.6.0: a new large-eddy simulation model in JAX with ... - GMD, truy cập vào tháng 2 23, 2026, https://gmd.copernicus.org/articles/19/1103/2026/

Adaptive Algorithms for 3D Reconstruction and Motion Estimation, truy cập vào tháng 2 23, 2026, https://www.vis.uni-stuttgart.de/abteilungen/computer_vision/publikationen/Maurer_Adaptive_algorithms_for_3D_reconstruction_and_motion_estimation_compressed_version.pdf

snnTrans-DHZ: A Lightweight Spiking Neural Network Architecture, truy cập vào tháng 2 23, 2026, https://arxiv.org/html/2504.11482v1

Class-imbalanced datasets | Machine Learning, truy cập vào tháng 2 23, 2026, https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets

Balanced MSE for Imbalanced Visual Regression, truy cập vào tháng 2 23, 2026, https://www.computer.org/csdl/proceedings-article/cvpr/2022/694600h916/1H0OkGgtIuQ

Balanced MSE for Imbalanced Visual Regression - CVF Open Access, truy cập vào tháng 2 23, 2026, https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.pdf

A Weighted Mean Square Error Technique to Train Deep Belief, truy cập vào tháng 2 23, 2026, https://ijssst.info/Vol-19/No-6/paper14.pdf

Research on Imbalanced Data Regression Based on Confrontation, truy cập vào tháng 2 23, 2026, https://www.mdpi.com/2227-9717/12/2/375

Deep Learning-Based Super-Resolution of Digital Elevation Models, truy cập vào tháng 2 23, 2026, https://eartharxiv.org/repository/object/4639/download/9268/

(PDF) Deep Imbalanced Multi-Target Regression: 3D Point Cloud, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/publication/397700613_Deep_Imbalanced_Multi-Target_Regression_3D_Point_Cloud_Voxel_Content_Estimation_in_Simulated_Forests

What is a Loss Function? Complete Guide - Articsledge, truy cập vào tháng 2 23, 2026, https://www.articsledge.com/post/loss-function

Loss Functions and Metrics in Deep Learning - arXiv, truy cập vào tháng 2 23, 2026, https://arxiv.org/html/2307.02694v5

Review of Methods for Handling Class-Imbalanced in Classification, truy cập vào tháng 2 23, 2026, https://arxiv.org/pdf/2211.05456

Advanced Class Imbalance Handling: From Basics to Super-Advanced, truy cập vào tháng 2 23, 2026, https://medium.com/@adnan.mazraeh1993/advanced-class-imbalance-handling-from-basics-to-super-advanced-65722f59c21b

Exploiting Digital Surface Models for Inferring Super-Resolution for, truy cập vào tháng 2 23, 2026, https://ieeexplore.ieee.org/iel7/36/9633014/09903066.pdf

Exploiting Digital Surface Models for Inferring Super-Resolution for, truy cập vào tháng 2 23, 2026, https://superworld.cyens.org.cy/papers/SRR_IEEE_TGRS_final.pdf

Balanced MSE for Imbalanced Visual Regression - Semantic Scholar, truy cập vào tháng 2 23, 2026, https://www.semanticscholar.org/paper/Balanced-MSE-for-Imbalanced-Visual-Regression-Ren-Zhang/d2a696b0803c227bc9bc5ff764dca74aadbf4b13

Deep Monocular Depth Estimation via Integration of Global and, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/publication/325159259_Deep_Monocular_Depth_Estimation_via_Integration_of_Global_and_Local_Predictions

[2203.16427] Balanced MSE for Imbalanced Visual Regression - arXiv, truy cập vào tháng 2 23, 2026, https://arxiv.org/abs/2203.16427

BALANCING MSE AGAINST ABRUPT CHANGES - OpenReview, truy cập vào tháng 2 23, 2026, https://openreview.net/pdf/d5635b8134c31f9037bee8b77c395dc066ddb161.pdf

Dist Loss: Enhancing Regression in Few-Shot Region through, truy cập vào tháng 2 23, 2026, https://openreview.net/forum?id=YeSxbRrDRl

Improve Representation for Imbalanced Regression through ... - CVPR, truy cập vào tháng 2 23, 2026, https://cvpr.thecvf.com/virtual/2025/poster/33100

Deep Learning Function Implications For Handwritten Character, truy cập vào tháng 2 23, 2026, https://bsj.uobaghdad.edu.iq/cgi/viewcontent.cgi?article=4876&context=home

[PDF] Automated Focal Loss for Image based Object Detection, truy cập vào tháng 2 23, 2026, https://www.semanticscholar.org/paper/Automated-Focal-Loss-for-Image-based-Object-Weber-F%C3%BCrst/3ba721988a5eddcc5a4dc0382fc45504d77f5cbc

Automated Focal Loss for Image based Object Detection, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/publication/348367999_Automated_Focal_Loss_for_Image_based_Object_Detection

A multimodal learning and simulation approach for perception in, truy cập vào tháng 2 23, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12886962/

Vision-Based Autonomous Vehicle Systems Based on Deep Learning, truy cập vào tháng 2 23, 2026, https://www.mdpi.com/2076-3417/12/14/6831

3D Point Cloud Voxel Content Estimation in Simulated Forests - arXiv, truy cập vào tháng 2 23, 2026, https://arxiv.org/html/2511.12740v1

Gradient-Domain Loss: Methods and Applications - Emergent Mind, truy cập vào tháng 2 23, 2026, https://www.emergentmind.com/topics/gradient-domain-loss

Direction-aware multi-scale gradient loss for infrared and visible, truy cập vào tháng 2 23, 2026, https://arxiv.org/html/2510.13067v1

Monocular 3D Reconstruction of Articulated Shapes with Weak, truy cập vào tháng 2 23, 2026, https://escholarship.org/content/qt8pm0d5wh/qt8pm0d5wh_noSplash_a6accb40f6dfbf9ca54678e0d9d79306.pdf?t=sawi98

Generative City Photogrammetry from Extreme Off-Nadir Satellite, truy cập vào tháng 2 23, 2026, https://arxiv.org/html/2512.07527v1

Low Complexity Networks and Edge Enhancement for Monocular, truy cập vào tháng 2 23, 2026, https://events.iist.ac.in/phd/thesis/SC16D007%20FT.pdf

LoD-2 Building Reconstruction from Satellite Imagery using Spatial, truy cập vào tháng 2 23, 2026, https://elib.dlr.de/218534/1/3rd_Journal_Paper.pdf

Normal Integration: A Survey - Computer Vision Group, truy cập vào tháng 2 23, 2026, https://cvg.cit.tum.de/_media/spezial/bib/jmiv_integration_1.pdf

(PDF) Normal Integration: A Survey - ResearchGate, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/publication/319896849_Normal_Integration_A_Survey

TOSD: A Hierarchical Object-Centric Descriptor Integrating Shape, truy cập vào tháng 2 23, 2026, https://www.mdpi.com/1424-8220/25/15/4614

Detection and Utilization of the Information Potential of Airborne, truy cập vào tháng 2 23, 2026, https://d-nb.info/1372180591/34

object detection in airborne lidar data for improved solar radiation, truy cập vào tháng 2 23, 2026, https://www.isprs.org/proceedings/XXXVIII/3-W8/papers/p69.pdf

New Tools or Trends for Large-Scale Mapping and 3D Modelling, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/profile/Fayez-Tarsha-Kurdi/publication/381310798_New_Tools_or_Trends_for_LargeScale_Mapping_and_3D_Modelling/links/66680806de777205a323932a/New-Tools-or-Trends-for-LargeScale-Mapping-and-3D-Modelling.pdf

3D Reconstruction of Old Tower Based on Airborne LiDAR of UAV, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/publication/388287100_3D_Reconstruction_of_Old_Tower_Based_on_Airborne_LiDAR_of_UAV

DIST: Rendering Deep Implicit Signed Distance Function ... - Microsoft, truy cập vào tháng 2 23, 2026, https://www.microsoft.com/en-us/research/wp-content/uploads/2020/06/DIST.pdf

DeepSDF: Learning Continuous Signed Distance Functions for, truy cập vào tháng 2 23, 2026, https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf

Information Theoretic Active Exploration in Signed Distance Fields, truy cập vào tháng 2 23, 2026, https://existentialrobotics.org/ref/Saulnier_ActiveMapping_ICRA20.pdf

Signed Distance Fields: A Natural Representation for Both Mapping, truy cập vào tháng 2 23, 2026, https://helenol.github.io/publications/rss_2016_workshop.pdf

DreamUp3D: Object-Centric Generative Models for Single-View 3D, truy cập vào tháng 2 23, 2026, https://arxiv.org/html/2402.16308v1

(PDF) Deep Learning on Object-Centric 3D Neural Fields, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/publication/382354378_Deep_Learning_on_Object-centric_3D_Neural_Fields

Predicting Sharp and Accurate Occlusion Boundaries in Monocular, truy cập vào tháng 2 23, 2026, https://michaelramamonjisoa.github.io/projects/DisplacementFields

Color bit depth info : r/StableDiffusion - Reddit, truy cập vào tháng 2 23, 2026, https://www.reddit.com/r/StableDiffusion/comments/1bgbkce/color_bit_depth_info/

Predicting High-precision Depth on Low-Precision Devices Using, truy cập vào tháng 2 23, 2026, https://arxiv.org/html/2405.14024v2

Predicting High-precision Depth on Low-Precision Devices Using, truy cập vào tháng 2 23, 2026, https://icml.cc/virtual/2025/poster/44496

Hilbert Space Filling Curve Based Scan-Order for Point Cloud, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/publication/361695369_Hilbert_Space_Filling_Curve_Based_Scan-order_for_Point_Cloud_Attribute_Compression

Predicting High-precision Depth on Low-Precision Devices ... - GitHub, truy cập vào tháng 2 23, 2026, https://raw.githubusercontent.com/mlresearch/v267/main/assets/uss25a/uss25a.pdf

An Improved Boundary-Aware Perceptual Loss for Building ... - MDPI, truy cập vào tháng 2 23, 2026, https://www.mdpi.com/2072-4292/12/7/1195

(PDF) An Improved Boundary-Aware Perceptual Loss for Building, truy cập vào tháng 2 23, 2026, https://www.researchgate.net/publication/340533628_An_Improved_Boundary-Aware_Perceptual_Loss_for_Building_Extraction_from_VHR_Images

Edge-Aware Normalized Attention for Efficient and Detail-Preserving, truy cập vào tháng 2 23, 2026, https://arxiv.org/html/2509.14550v1

Supervised Learning With Perceptual Similarity for Multimodal Gene, truy cập vào tháng 2 23, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC8355627/

Deep Learning based Super Resolution of Urban Digital Surface, truy cập vào tháng 2 23, 2026, https://elib.dlr.de/204464/1/NallanukalaKT-MTReport-compress.pdf

Edge-preserving and scale-dependent properties of total variation, truy cập vào tháng 2 23, 2026, https://ww3.math.ucla.edu/camreport/cam00-38.pdf

outlier robust corner-preserving methods for - Euclid, truy cập vào tháng 2 23, 2026, https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-1/Outlier-robust-corner-preserving-methods-for-reconstructing-noisy-images/10.1214/009053606000001109.pdf


| Imbalanced Regression Loss Paradigm | Core Mathematical Mechanism | Advantage for nDSM Generation | Potential Implementation Drawback |
|---|---|---|---|
| Balanced MSE (BMSE) | Statistically converts prediction probabilities using label distribution priors to enforce uniform generalization.22 | Provides a theoretically sound, uniform generalization across all height bins without exploding gradients; highly stable.22 | Requires pre-calculation of dataset height distributions and the establishment of binning heuristics.37 |
| Focal Regression Loss | Applies a dynamic scaling factor based directly on the current magnitude of the error to down-weight easy predictions.40 | Automatically down-weights the easily predicted ground pixels; requires absolutely no prior dataset density analysis.40 | Highly sensitive to the hyperparameter tuning of the focusing exponent; can potentially overfit to noisy outliers in the DSM.41 |
| Density-Based Relevance | Reweights pixel loss based on the inverse Kernel Density Estimate of the target height.44 | Smoothly penalizes errors based on exact geometric rarity, accounting for both sample size and uniform deviation.44 | High computational overhead if density is computed dynamically; static KDE may fail if the batch variance is unusually high.44 |


| Geometric Loss Component | Mathematical Target | Optimization Effect on nDSM |
|---|---|---|
| Direction-Aligned Gradient | and  independently with sign.46 | Defines exactly which side is ground and which is object based on the polarity of the step.46 |
| Laplacian Regularization | using an  sparsity norm.48 | Forces the height transition to occur over a single pixel rather than a smoothed gradient.48 |
| Surface Normal Consistency | Cosine distance of the cross-product of spatial derivatives.51 | Enforces 90-degree orthogonal walls and completely flat ground planes, separating objects perfectly.50 |


| Training Objective Phase | Active Loss Components | Primary Goal of Phase | Impact on ControlNet Architecture |
|---|---|---|---|
| Phase 1: Warm-up | Balanced MSE + L1 Pixel Loss | Establish base elevation values and overcome massive ground-class imbalance.22 | Safely activates zero-convolutions without triggering gradient explosion in the adapter.11 |
| Phase 2: Structural | Direction-Aligned Gradient + Boundary Perceptual | Align height transitions with true spatial footprints; define step directions.46 | Forces U-Net attention layers to respect object boundaries in the latent space.70 |
| Phase 3: Geometric | Laplacian + Surface Normal + SDF | Enforce orthogonal walls, perfectly flat roofs, and definitive interior/exterior separation.48 | Crystallizes the final micro-textures; sharpens Heaviside discontinuities into true step functions.18 |
