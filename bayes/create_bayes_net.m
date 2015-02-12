function[net] = create_bayes_net()
    
N = 4; 
dag = zeros(N,N);
C = 1; S = 2; R = 3; W = 4;
dag(C,[R S]) = 1;
dag(R,W) = 1;
dag(S,W)=1;

discrete_nodes = 1:N;
node_sizes = 2*ones(1,N);

% bnet = mk_bnet(dag, node_sizes);
bnet = mk_bnet(dag, node_sizes, 'names', {'cloudy','S','R','W'}, 'discrete', 1:4);

CPT = reshape([1 0.1 0.1 0.01 0 0.9 0.9 0.99], [2 2 2]);
bnet.CPD{C} = tabular_CPD(bnet, C, [0.5 0.5]);
bnet.CPD{R} = tabular_CPD(bnet, R, [0.8 0.2 0.2 0.8]);
bnet.CPD{S} = tabular_CPD(bnet, S, [0.5 0.9 0.5 0.1]);
bnet.CPD{W} = tabular_CPD(bnet, W, [1 0.1 0.1 0.01 0 0.9 0.9 0.99]);

G = bnet.dag;
draw_graph(G);


end


