#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
using namespace std;

// Функция для нормализации данных
vector<vector<double>> normalizeData(const vector<vector<double>>& data) {
    int n = data.size();
    if (n == 0) return {}; // Возвращаем пустой вектор, если данные пустые.

    int m = data[0].size();
    vector<vector<double>> normalizedData(n, vector<double>(m));
    vector<double> means(m, 0.0);
    vector<double> stdevs(m, 0.0);

    // Вычисление средних значений
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            means[j] += data[i][j];
        }
    }
    for (int j = 0; j < m; ++j) {
        means[j] /= n;
    }

    // Вычисление стандартных отклонений
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            stdevs[j] += pow(data[i][j] - means[j], 2);
        }
    }
    for (int j = 0; j < m; ++j) {
        stdevs[j] = (stdevs[j] / n > 0) ? sqrt(stdevs[j] / n) : 1;  // Обработка 0 отклонения
    }

    // Нормализация
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            normalizedData[i][j] = (stdevs[j] != 0) ? (data[i][j] - means[j]) / stdevs[j] : 0;
        }
    }
    return normalizedData;
}

bool equal_weights(const vector<double>& m1, const vector<double>& m2){
        if (m1.size() != m2.size()) return false;
        for (int i = 0; i < m1.size(); i++){
                if (abs(m1[i] - m2[i])>0.001) return false;
            };
        return true;
    };

double get_accuracy(vector<int> m1, vector<int> m2){
    double accuracy;
    
    for (int i=0;i<m1.size();i++){
        if (m1[i]==m2[i]) accuracy++;
    }

    return accuracy/m1.size();
}

class HebbianPerceptron {
private:
    vector<double> weights;
    double threshold;
    double learningRate;

public:
    HebbianPerceptron(double learningRate = 0.1, double threshold = 0.0) :
        weights{0.0, 0.0}, threshold(threshold), learningRate(learningRate) {} // Размер вектора весов - 2

    int predict(const vector<double>& input) const {
        if(input.size() != weights.size()) {
            cout<<input.size()<<" "<<weights.size();
            cerr << "Error: Input vector size mismatch." << endl;
            return -1; // Return -1 to signal an error.
        }

        double sum = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            sum += weights[i] * input[i];
        }
        return (sum >= threshold) ? 1 : 0;
    }


    void train(const vector<pair<vector<double>, int>>& data, int maxEpochs) {
        weights.assign(2, 0.0); // Обнуляем веса перед обучением
        int epochCount = 0;
        vector<double> prevWeights = weights; // Use a vector for prev weights.

        while (epochCount < maxEpochs) {
            bool allCorrect = true;
            vector<double> weightChanges(weights.size(), 0.0);

            for (const auto& sample : data) {
                const vector<double>& input = sample.first;
                int target = sample.second;
                int prediction = predict(input);
                if (prediction == -1) return; // Handle error from predict

                if (prediction != target) {
                    allCorrect = false;
                    for (size_t i = 0; i < weights.size(); ++i) {
                        weightChanges[i] += learningRate * (target - prediction) * input[i];
                    }
                }
            }
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] += weightChanges[i];
            }
            for (double i=0;i<weights.size();i++) cout<<weights[i]<<" ";  
            cout<<endl;        
            // Check for convergence (using vector comparison)
            if (equal_weights(weights, prevWeights)) {
                cout << "Successfully for " << epochCount << " epochs." << endl;
                return;
            }
            prevWeights = weights;
            

            if (allCorrect) {
                cout << "Success " << epochCount << " epochs." << endl;
                return;
            }
            epochCount++;
        }
        cout << "Failed with " << maxEpochs << " epochs." << endl;
    }
};

int main() {
    /*
    vector<vector<double>> inputs;
    vector<int> targets;
    ifstream file("learn_data.txt");

    string line;
    int i = 0;
    while(getline(file, line)){
        stringstream ss(line);
        vector<double> features;
        double val;
        while (ss >> val) {
            features.push_back(val);
        }
        if (i<50){
            targets.push_back(0);
        }
        else{
            targets.push_back(1);
        }
        inputs.push_back(features); 
        i++;  
    }
    file.close();
    cout<<inputs.size();*/
    // Пример линейно разделимых данных (измените, если нужно)
    
    // Генерация линейно разделимого набора данных
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0); // Диапазон для генерации случайных чисел
    int num_samples = 60;

    vector<vector<double>> inputs, cl1, cl2;
    vector<int> targets;

    for (int i = 0; i < num_samples; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        int target;
        if (x + y > 10) { // Линейное разделение
            target = 0;
            cl1.push_back({x, y});
        } else {
            target = 1;
            cl2.push_back({x, y});
        }
        //cout<<x<<", "<<y<<":"<<target<<endl;
        inputs.push_back({x, y});
        targets.push_back(target);
    }
    
    vector<vector<double>> normalizedInputs = normalizeData(inputs); // Нормализуем данные

    vector<pair<vector<double>, int>> data;
    for (size_t i = 0; i < normalizedInputs.size(); ++i) {
        data.push_back({normalizedInputs[i], targets[i]});
    }
    
    HebbianPerceptron perceptron;
    perceptron.train(data, 1000);

    ofstream file;
    file.open("lear.txt");
    for (int i=0;i<cl1.size();i++)
    {   
        for (int j=0;j<cl1[0].size();j++){
            file<<cl1[i][j]<<" ";
        }
        file<<endl;
    }
    file<<endl;
    for (int i=0;i<cl2.size();i++)
    {   
        for (int j=0;j<cl2[0].size();j++){
            file<<cl2[i][j]<<" ";
        }
        file<<endl;
    }
    file.close();
    cout<<endl<<cl1.size()<<" "<<cl2.size();
    cout<<endl;
/*
    vector<vector<double>> norm, norma;
    int prediction;
    for (int i = 0;i<20;i++){
        double x = dis(gen);
        double y = dis(gen);
        norm.push_back({x, y});
    }
    norma = normalizeData(norm);
    for (int i=0;i<25;i++){
        prediction = perceptron.predict({norma[i][0], norma[i][1]});
        cout<<"("<<norm[i][0]<<", "<<norm[i][1]<<"): "<<prediction<<" class"<<endl;
    }
*/

    cout<<endl;
    cout<<endl;
    vector<int> m1, m2;
   for (size_t i = 0; i < normalizedInputs.size(); ++i) {
        int prediction = perceptron.predict(normalizedInputs[i]);
        m1.push_back(prediction);
        m2.push_back(targets[i]);
        //cout << "Example: " << i + 1 << ", Predicted class: " << prediction
             //<< ", True class: " << targets[i] << endl;
    }
    //cout<<endl<<get_accuracy(m1,m2);
    return 0;
}