load("/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsCNMFE/20-Jul_15_08_57.mat");
%load("/home/user/Documents/single_session/preprocessed/preprocessedMovie_source_extraction/frames_1_11998/LOGS_03-May_15_58_06/03-May_16_57_40.mat");
% neuron.A is a 2D matrix with the dimension (# pixels X # neurons)
% neuron.C is also a 2D matrix with the dimension (# neurons X # frames)

% You can visualize the neuron shape using the following commands (traces
% only second one)
% i=1;
% neuron.image(neuron.A(:, i)); 
% neuron.viewNeurons(i); 

% visualize all
% neuron.viewNeurons([]); 

% save neurons
% neuron.save_neurons()

% save workspace
%neuron.save_workspace()

% Usually, we want to represent each neuron's shape as a 2D matrix, instead 
% of a single column vector. To get a 2D representation of a neuron shape, 
% we can use following command:
% ai = neuron.reshape(neuron.A(:, i), 2)

% or you can convert the 2D matrix neuron.A into a 3D matrix A and A(:, :, i) 
% represents the spatial shape of the i-th neuron:
% A = neuron.reshape(neuron.A, 2); 

% sizes
% sz = size(neuron.A);
% szdim = size(neuron.A,dim);
% disp(size(neuron.A(:,1)));

% coordinates
% coords = neuron.show_contours();
coords = neuron.get_contours();

A = neuron.reshape(neuron.A, 2); 

% neuron.C is also a 2D matrix with the dimension (# neurons X # frames)
C = neuron.C;
% disp(C)

%get cellmap
cellmap = neuron.Cn;
disp(coords);
%save matfile
save('/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsCNMFE/traces_synthetic.mat','C');
save('/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsCNMFE/binary_masks_synthetic.mat','A');
save('/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsCNMFE/cellmap_synthetic.mat','cellmap');
save('/home/user/Documents/synthetic_datasets/sequence_10/extracted/resultsCNMFE/coords_synthetic.mat','coords');
