function y = sigmoid_prime(z)
    y = sigmoid(z).*(1-sigmoid(z));
end