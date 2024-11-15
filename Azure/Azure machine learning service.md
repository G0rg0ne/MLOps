Core elements of the AMLS : 
![[Pasted image 20240823154736.png]]
* Notebooks 
* AutoML : Completely automated process to build and train an ML model
* Designer : Visual drag and drop designer to construct end to end ML pipelines
* Datasets: Data that you upload which will be used for training
* Experiments : when you run a training job they are detailed here
* Pipelines : ML workflows you have built, or you have used in the Designer
* Models : a model registry containing trained models that can be deployed
* Endpoints: When you deploy a model its hosted on an accessible endpoint eg. REST API
* Compute: the underlying computing instances used to for notebooks, training, inference
* Environments : a reproducible Python environment for machine learning experiments
* Datastores : have humans with ML-assisted labeling to label your data for supervised learning
* Linked services: external services you can connect to the workspace eg.Azure synapse
## Compute : 
### Types of computes : 
* Compute Instances : Development workstations that data scientists can use to work with data and models
* Compute Clusters : Scalable clusters of virtual machines for on-demand processing of experiment code.
* Inference Clusters : Deployment targets for predictive services that use your trained models
* Attached Compute : Links to existing Azure compute resources, such as Virtual Machines or Azure Databricks clusters.
### Data labeling : 
Create data labeling jobs to prepare your Ground truth for supervised learning : 
- Human-in-the-loop labeling
- Machine-learning-assisted data labeling

### Data stores  :
- Azure blob storage : data is stored as objects, distributed across many machines.
- Azure file share : a mountable file via SMB and NFS protocols
- Azure data lake storage (GEN2) : Azure blob storage designed for vasts amount of data for big data analytics
- Azure SQL database : Full-managed MS SQL relational database
- Azure postgres database : Open-source relational database
- Azure MySQL database : Open-source relational database
- 