function net = vp_detect_model(model, drop6, drop7)
    % load a model and upgrade it to MatConvNet current version.
    net = load(model);

    % remove the softmax loss, as it is used for learning phase only
    net.layers(end) = [];
    net.layers{end+1} = struct('type', 'softmax');

    % remove the drop layers, also only used for learning
    net.layers = [net.layers(1:(drop6-1)) net.layers((drop6+1):(drop7-1)) net.layers((drop7+1):end)];

    % clear up, fill in defualts
    net = vl_simplenn_tidy(net) ;

    % show the layers in the model
    vl_simplenn_display(net, 'inputSize', [224 224 3 50]);
end