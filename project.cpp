#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <cmath>
#include <algorithm>
using namespace std;

int class_k;
vector<int> class_k_elements;
int k_izm;

struct ClassParams{
    int k_izm;
    vector<double> min_ranges;
    vector<double> max_ranges;

    ClassParams(int k, vector<double> min, vector<double> max){
        k_izm = k;
        min_ranges = min;
        max_ranges = max;
    };
};

class ClassParamsRepository
{
    public:
    vector<ClassParams> repository;

    // Конструктор репозитория классов
    ClassParamsRepository(string generating_struct_file)
    {

        // Открываем файл для считывания
        ifstream file1(generating_struct_file); 
        string s;
        // Считываем количество классов
        getline(file1, s);
        class_k = stoi(s);

        // Считываем количество элементов для каждого класса
        getline(file1, s);
        stringstream ss(s);
        string k;
        while (ss >> k){
            class_k_elements.push_back(stoi(k));
        };

        // Считываем количество измерений
        getline(file1,s);
        k_izm = stoi(s);

        // Считываем ограничения классов, создаем структуры и заносим их в репозиторий
        
        for (int i = 0; i < class_k; i++){
            getline(file1,s);
            int n = 0;
            vector<double> min_points_ranges{};
            vector<double> max_points_ranges{};
            string p = "";
            stringstream ss(s);
            while (ss >> p){
                if (n < k_izm) min_points_ranges.push_back(stoi(p));
                else max_points_ranges.push_back(stoi(p));
                n++;
            };
            // Заносим в репозиторий
            repository.push_back(ClassParams(k_izm, min_points_ranges, max_points_ranges));
        };
        file1.close();
    }   
};

struct points_for_1Class
{
    vector <vector <double>> point_coordinates;
    vector<double> v;
    
    points_for_1Class(string filename,ClassParams strukt, int k_el){
        for (int i = 0; i < k_el; i++){
            v.clear();
            for (int j = 0; j < strukt.k_izm ; j++) {
                int start = strukt.min_ranges[j];
                int end = strukt.max_ranges[j]; 
                random_device rd;
                mt19937 gen(rd());
                uniform_real_distribution<double> dis(start,end);
                double random_double=dis(gen);
                v.push_back(random_double);
            };
            point_coordinates.push_back(v);
        };
        // Отправляем данные точек в файл
        ofstream file;
        file.open(filename, ios::app);
        for (int i = 0; i < k_el; i++) {
            for (int j = 0; j < strukt.k_izm; j++) {
                file << point_coordinates[i][j]<<" ";
            };
            file << endl;
        };
        file.close();
    };
};

class ClassAll_Points{
    public: 
    vector<points_for_1Class> points_repos;

    void addClass(points_for_1Class struk){
        points_repos.push_back(struk);}
};

class kash
{
    public:
    vector<vector<double>> matrica, y;
    int h = 1;

    kash(vector<vector<double>> m1,vector<vector<double>> m2)
    {
        for (int i=0;i<(m1.size() + m2.size());i++) y.push_back({1});

        for (int i=0;i<m1.size();i++) m1[i].push_back(1);
        for (int i=0;i<m2.size();i++) m2[i].push_back(1);

        for (int i = 0; i < size(m2);i++){   
                for (int j = 0; j < size(m2[0]);j++){
                    m2[i][j]=-m2[i][j];}}
        
        for (int i=0;i<m1.size();i++) matrica.push_back(m1[i]);
        for (int i=0;i<m2.size();i++) matrica.push_back(m2[i]);
    };

    void printMatr(vector <vector <double>> matr){
        for (int i = 0; i < size(matr);i++){   
                for (int j = 0; j < size(matr[0]);j++){
                    cout<<matr[i][j]<<" ";
            }
            cout<<endl;
            }
            return;
    }

    vector <vector <double>> matrXmatr(const vector <vector <double>>& matr1,const vector <vector <double>>& matr2)
    {
        vector <vector <double>> resultM;
        if (matr1[0].size()==matr2.size())
        {
            vector <vector <double>> m1=matr1;
            vector <vector <double>> m2=matr2;
            vector <double> part;
            int sz1=size(m1);
            int sz2=size(m2);
            for (int i = 0; i < sz1;i++){   
                for (int j = 0; j < size(m2[0]);j++){
                    double s=0;
                    for (int l = 0; l < sz2;l++){
                        s=s+m1[i][l]*m2[l][j];}
                    part.push_back(s);}
                resultM.push_back(part);
                part.clear();}
            return resultM;
        }
        cout<<"Incorrect sizes: can`t multiply"<<endl;
        return resultM;
    }

    vector <vector <double>> tranMatr(vector <vector <double>> matr)
    {
        int rows = size(matr);
        int cols = size(matr[0]);
        vector <vector <double>> tMatr;
        vector <double> part;
        for (int i = 0; i < cols;i++){  
                for (int j = 0; j < rows;j++){
                    part.push_back(matr[j][i]);}
            tMatr.push_back(part);
            part.clear();}
        return tMatr;
    }

    vector<vector<double>> reverseMatr(vector<vector<double>> matr, double d){

        for (int i = 0; i < size(matr);i++){   
                for (int j = 0; j < size(matr[0]);j++)
                {
                    matr[i][j]=matr[i][j]/d;
                }
            }
            return matr;
    }

    vector<vector<double>> getMinor(const vector<vector<double>>& matrix, int p, int q, int n) {
        vector<vector<double>> minor(n - 1, vector<double>(n - 1));
        int row = 0, col = 0;

        for (int i = 0; i < n; i++) {
            if (i == p)
                continue;
            col = 0;
            for (int j = 0; j < n; j++) {
                if (j == q)
                    continue;
                minor[row][col] = matrix[i][j];
                col++;
            }
            row++;
        }
        return minor;
    }
    // Рекурсивная функция для вычисления определителя
    double determinant(const vector<vector<double>>& matrix, int n) {
        if (n == 1)
            return matrix[0][0];

        double det = 0;
        int sign = 1;

        for (int f = 0; f < n; f++) {
            vector<vector<double>> minor = getMinor(matrix, 0, f, n);
            det += sign * matrix[0][f] * determinant(minor, n - 1);
            sign = -sign;
        }

        return det;
    }
    // Функция для вычисления минора матрицы
    double minor(const vector<vector<double>>& matrix, int row, int col) {
        int size = matrix.size();
        vector<vector<double>> submatrix(size - 1, vector<double>(size - 1));

        int sub_i = 0;
        for (int i = 0; i < size; ++i) {
            if (i == row) continue;
            int sub_j = 0;
            for (int j = 0; j < size; ++j) {
                if (j == col) continue;
                submatrix[sub_i][sub_j] = matrix[i][j];
                ++sub_j;
            }
            ++sub_i;
        }
        
        // Вычисляем детерминант минорной матрицы
        double det = 0.0;
        if (submatrix.size() == 2) {
            det = submatrix[0][0] * submatrix[1][1] - submatrix[0][1] * submatrix[1][0];
        } else {
            // Рекурсивный вызов для больших подматриц
            for (int i = 0; i < submatrix.size(); ++i) {
                det += ((i % 2 == 0) ? 1 : -1) * submatrix[0][i] * minor(submatrix, 0, i);
            }
        }
        return det;
    }
    // Функция для вычисления матрицы алгебраических дополнений
    vector<vector<double>> cofactorMatrix(const vector<vector<double>>& matrix) {
        int size = matrix.size();
        vector<vector<double>> cofactors(size, vector<double>(size));

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                // Вычисляем минор для элемента (i, j)
                double minorValue = minor(matrix, i, j);

                // Вычисляем алгебраическое дополнение, учитывая знак
                cofactors[i][j] = ((i + j) % 2 == 0 ? 1 : -1) * minorValue;
            }
        }
        return cofactors;
    }
    
    vector<vector<double>> heavyside_f(vector<vector<double>> v){
        for (int i=0;i<v.size();i++) for (int j=0;j<v[0].size();j++){
            if (v[i][j] < 0) v[i][j] = 0;
            else v[i][j] = 1;
        } 
        return v;
    }
    
    bool iszero(vector <vector <double>> a) {
    bool b = true;
    for (int i = 0; i < a.size(); i++) {
    if (a[i][0] != 0) b = false;
    }
    return b;
    }
    bool negative(vector<vector<double>> vw){
        for (int i = 0; i < size(vw);i++){   
                for (int j = 0; j < size(vw[0]);j++){
                    if (vw[i][j] < 0) return true;}
            }
            return false;
    }
    vector<vector<double>> v_PLUS_v(vector<vector<double>> v1, vector<vector<double>> v2){
        for (int i=0;i<v1.size();i++) for (int j=0;j<v1[0].size();j++) v1[i][j] = v1[i][j] + v2[i][j];
        return v1;
    }
    vector<vector<double>> v_MINUS_v(vector<vector<double>> v1,vector<vector<double>> v2){
        for (int i=0;i<v1.size();i++) for (int j=0;j<v1[0].size();j++) v1[i][j] = v1[i][j] - v2[i][j];
        return v1;
    }
    vector<vector<double>> v_X_v(vector<vector<double>> v, double n){
        for (int i=0;i<v.size();i++) for (int j=0;j<v[0].size();j++) v[i][j] = v[i][j] * n;
        return v;
    }
    
    void psevdo_solution(){
        vector<vector<double>> mXt = matrXmatr(tranMatr(matrica),matrica);
        int n = mXt.size();
        vector<vector<double>> psevdoobr = matrXmatr(reverseMatr(cofactorMatrix(mXt),determinant(mXt,n)),tranMatr(matrica));
        vector<vector<double>> w = matrXmatr(psevdoobr,y);
        vector<vector<double>> vw = matrXmatr(matrica,w);

        //Поиск псевдорешения
        while (negative(vw)){
        if ((iszero(heavyside_f(v_MINUS_v(vw, y)))) and (not iszero(v_MINUS_v(vw, y)))) {
        cout<<"Classes are not separable!";
        return;
        }
        vector<vector<double>> w = matrXmatr(psevdoobr,y);
        vector<vector<double>> vw = matrXmatr(matrica,w);
        y = v_PLUS_v(y, v_X_v(heavyside_f(v_MINUS_v(vw, y)), h));
        }
        ofstream file;
        file.open("kashyap_lines.txt", ios::app);
        for (int i=0;i<w.size();i++) file<<" "<<w[i][0];
        file<<endl;
        file.close();
        return;
    }
};

struct metricks{
    vector<double> centroid(vector<vector<double>> m){
        vector<double> avg(k_izm);
        for (int i=0;i<k_izm;i++) for (int j=0;j<m.size();j++) avg[i]=avg[i]+m[j][i];
        for (int i=0;i<k_izm;i++) avg[i] = avg[i] / m.size();
        return avg;
    }
    void evklidova(vector<vector<double>> m1,vector<vector<double>> m2){
        vector<double> c1=centroid(m1);
        vector<double> c2=centroid(m2);
        double s = 0;
        for (int i = 0;i<k_izm;i++) s=s+(c1[i]-c2[i])*(c1[i]-c2[i]);
        cout<<"Evklidov Metrick = "<<sqrt(s)<<endl;
        return;
    }
    void manhat(vector<vector<double>> m1, vector<vector<double>> m2){
        vector<double> c1=centroid(m1);
        vector<double> c2=centroid(m2);
        double s = 0;
        for (int i = 0;i<k_izm;i++) s=s+abs(c1[i]-c2[i]);
        cout<<"Manhat Metrick = "<<s<<endl;
        return;
    }
};

struct create_clusters
{
    double evklid(vector<double> m1,vector<double> m2){
        double s = 0;
        for (int i = 0;i<k_izm;i++) s=s+(m1[i]-m2[i])*(m1[i]-m2[i]);
        return sqrt(s);
    }
  // Плотность образов
    double find_density(vector<vector<double>> elements, int k_elem){
        double dens = 0;
        double h = 10;
        for (int i = 0; i < elements.size(); i++){
        dens = dens + (1 / pow(h, 2)) * (pow(h, 2) - pow(evklid(elements[k_elem], elements[i]), 2));
        };
        return dens;
    };

    vector<vector<double>> copy_vector(vector<vector<double>> vector_to_copy){
        vector<vector<double>> new_vector;
        vector<double> new_row;
        for (int i = 0; i < vector_to_copy.size(); i++){
        new_row = {};
        for (int j = 0; j < vector_to_copy[0].size(); j++){
        new_row.push_back(vector_to_copy[i][j]);
        };
        new_vector.push_back(new_row);
        };
        return new_vector;
    };

    vector<vector<double>> alg_proseivaniya(vector<vector<double>> elements, int k_elems){
        vector<vector<double>> result_points;
        vector<double> densities;
        for (int i = 0; i < elements.size(); i++){
        densities.push_back(find_density(elements, i));
        };
        vector<vector<double>> vec = copy_vector(elements);
        double max;
        int n;
        for (int i = 0; i < k_elems; i++){
            max = densities[0];
            n = 0;
            for (int j = 0; j < densities.size(); j++){
                if (densities[j] > max){
                    max = densities[j];
                    n = j;
                };
            };
            result_points.push_back(elements[n]);
            densities[n] = -1000000;
            };
        return result_points;
    };

    vector<double> find_new_max_point(vector<vector<double>> elements, vector<vector<double>> max_points){
        vector<double> distances_to_mp;
        vector<double> distances;
        double min;
        double max;
        int n;
        for (int i = 0; i < elements.size(); i++){
        distances_to_mp = {};
        for (int j = 0; j < max_points.size(); j++) distances_to_mp.push_back(evklid(elements[i], max_points[j]));
        
        min = distances_to_mp[0];
        for (int j = 0; j < distances_to_mp.size(); j++){
            if (distances_to_mp[j] < min) min = distances_to_mp[j];
        };
        distances.push_back(min);
        };

        n = 0;
        max = distances[0];
        for (int i = 0; i < distances.size(); i++){
            if (distances[i] > max){
                max = distances[i];
                n = i;
        };
        };
        return elements[n];
    };

    vector<vector<double>> alg_max_distance(vector<vector<double>> elements, int k_elements){
        vector<vector<double>> vec = copy_vector(elements);
        vector<vector<double>> result_points;
        result_points.push_back(vec[0]);
        for (int i = 0; i < k_elements - 1; i++){
        result_points.push_back(find_new_max_point(elements, result_points));
        };
        return result_points;
    };

    vector<vector<vector<double>>> create_clasters(vector<vector<double>> centres, vector<vector<double>> elems){
        vector<vector<vector<double>>> clasters(centres.size());
        int n;
        double min;
        for (vector<double> el : elems){
            min = evklid(el, centres[0]);
            n = 0;
            for (int i = 0; i < centres.size(); i++){
                if (evklid(el, centres[i]) < min){
                    min = evklid(el, centres[i]);
                    n = i;
                };
            };
            clasters[n].push_back(el);
        };
        return clasters;
    };

    vector<double> find_new_claster_centroid(vector<vector<double>> claster){
        if (claster.size() == 0) return {};
        vector<double> new_centroid;
        double sum;
        double avg;
        for (int i = 0; i < claster[0].size(); i++){
            sum = 0;
            for (int j = 0; j < claster.size(); j++) sum = sum + claster[j][i];
            avg = sum / claster.size();
            new_centroid.push_back(avg);
        };
        return new_centroid;
    };

    vector<vector<double>> create_new_centroids(vector<vector<vector<double>>> clasters){
        vector<vector<double>> new_centroids;
        for (vector<vector<double>> cl : clasters){
        new_centroids.push_back(find_new_claster_centroid(cl));
        };
        return new_centroids;
    };

    bool equal_matrices(vector<vector<double>> m1, vector<vector<double>> m2){
        if (m1.size() != m2.size()) return false;
        for (int i = 0; i < m1.size(); i++){
        for (int j = 0; j < m1[0].size(); j++){
        if (m1[i][j] != m2[i][j]) return false;
        };
        };
        return true;
    };

    vector<vector<vector<double>>> iterated_obrazy(vector<double> cent, vector<vector<double>> elems, double rad){
        vector<vector<vector<double>>> train(2);
        for (int i=0;i<elems.size();i++){
            if (evklid(elems[i], cent)<=rad){
                train[0].push_back(elems[i]);
            }
            else train[1].push_back(elems[i]);
        }
        return train;
        }

    vector<vector<double>> alg_FOREL(vector<vector<double>> elements, int r){
        vector<vector<double>> centers, clust;
        vector<double> c, point;
        vector<vector<vector<double>>> obrazy={clust, elements}; 
        
        while (obrazy[1].size()!=0){
            point = obrazy[1][0];
            obrazy = iterated_obrazy(point, obrazy[1], r);	
            clust = obrazy[0];
            c = find_new_claster_centroid(clust);
            centers.push_back(c);
        };
        return centers;
    }
        // функция вычисления дисперсии по одной компоненте
    double calculate_variance(vector<vector<double>> elements, int dimension) {
        if (elements.empty()) return 0;

        double sum = 0;
        for (vector<double> point : elements) {
            sum += point[dimension];
        }
        double mean = sum / elements.size();

        double sq_diff_sum = 0;
        double diff;
        for (vector<double> point : elements) {
            diff = point[dimension] - mean;
            sq_diff_sum += pow(diff, 2);
        }
        return sq_diff_sum / elements.size();
    }
     // функция разделения кластера
    vector<vector<vector<double>>> split_cluster(vector<vector<double>> elements, double threshold) {
        double variance_x = calculate_variance(elements, 0);
        double variance_y = calculate_variance(elements, 1);
        vector<vector<vector<double>>> result;

        int split_dimension = (variance_x > variance_y) ? 0 : 1; // Определяем ось для разделения

        if ( (split_dimension == 0 && variance_x > threshold) || (split_dimension == 1 && variance_y > threshold) ) {
            double mean;      
            double sum = 0;
            for (vector<double> point : elements) sum += point[split_dimension];
            mean = sum / elements.size();
            
            vector<vector<double>> cluster1, cluster2;
            for (vector<double> point : elements) {
                if (point[split_dimension] < mean) {
                    cluster1.push_back(point);
                } else {
                    cluster2.push_back(point);
                }
            }
            result.push_back(cluster1);
            result.push_back(cluster2);
            return result;
        } 
        else {
            return result; // Кластер не разделен
        }
    }

    vector<vector<double>> merge_clusters(vector<vector<double>> cluster1, vector<vector<double>> cluster2) {
        vector<vector<double>> res;
        for (int i = 0; i < cluster1.size(); i++) res.push_back(cluster1[i]);    
        for (int i = 0; i < cluster2.size(); i++) res.push_back(cluster2[i]);
        return res;
    }

    vector<vector<double>> iso_data(vector<vector<vector<double>>> clusters, int porog, int Nmin, int dmin){
        vector<vector<vector<double>>> last = clusters;
        vector<vector<vector<double>>> tmp = clusters;
        vector<vector<vector<double>>> next;
        vector<vector<vector<double>>> splitted;
        vector<vector<double>> res;
        
        //          Разделение
        while (last.size() != next.size()){
            last = tmp;
            next.clear();
            for (int i=0;i<last.size();i++){
                splitted = split_cluster(last[i], porog);
                if (splitted.size()){    
                    for (int j =0;j<splitted.size();j++) next.push_back(splitted[j]);
                }
                else{
                    next.push_back(last[i]);
                }
            }
            tmp = next;
        }

        //          Слияние
        cout<<tmp.size()<<endl;
        next.clear();
        vector<int> merged_indexes;
        double diam;
        vector<vector<double>> merged;
        vector<double> c1, c2;
        
        while (last.size() != next.size()){
            last = tmp;
            next.clear();
            merged_indexes.clear();
            for (int i=0;i<last.size()-1;i++){
                if (find(merged_indexes.begin(), merged_indexes.end(), i) == merged_indexes.end()){
                    for (int j=i+1;j<last.size();j++){
                        if (find(merged_indexes.begin(), merged_indexes.end(), j) == merged_indexes.end()){
                            c1 = find_new_claster_centroid(last[i]);
                            c2 = find_new_claster_centroid(last[j]);
                            diam = evklid(c1, c2);
                            if (diam < dmin){
                                merged = merge_clusters(last[i], last[j]);
                                next.push_back(merged);
                                merged_indexes.push_back(i);
                                merged_indexes.push_back(j);
                                break;
                            }
                        }
                    }
                }
            }
            for (int i=0;i<last.size();i++){
                if (find(merged_indexes.begin(), merged_indexes.end(), i) == merged_indexes.end()){
                    next.push_back(last[i]);
                }
            }
            tmp = next;
        }

        //      Удаление
        cout<<tmp.size()<<endl;
        next.clear();
        for (int i=0;i<tmp.size();i++){
            if (tmp[i].size()>=Nmin){
                next.push_back(tmp[i]);
            }
        }
        cout<<next.size()<<endl;
        ofstream file;
        file.open("iso_filtered_pts.txt");
        for (const auto& cluster : next) {
            for (const auto& point : cluster) {
                file << point[0] << " " << point[1] << endl;
            }
            file << endl;
        }
        file.close();
        for (int i=0;i<next.size();i++) res.push_back(find_new_claster_centroid(next[i]));
        return res;
    }
 
    void k_means(ClassAll_Points repos, int k){
        vector<vector<double>> all_elems;
        vector<vector<double>> centroids1, centroids2, centroids3, centroids4;
        int porog = 10;
        int Nmin = 10;
        int dmin = 6;

        for (int i = 0; i < repos.points_repos.size(); i++){
            for (vector<double> j : repos.points_repos[i].point_coordinates){
                all_elems.push_back(j);
            };
        };
        centroids1 = alg_proseivaniya(all_elems, k);
        centroids2 = alg_max_distance(all_elems, k);
        int radius = 25;
        centroids3 = alg_FOREL(all_elems,radius);
        ofstream file;
        file.open("forel_centroids.txt");
        file<<radius<<endl;
        for (int i=0;i<centroids3.size();i++){
            for (int j=0;j<centroids3[0].size();j++){
                file<<to_string(centroids3[i][j])<<" ";
            } 
            file<<endl;
        }
        file.close();
        
        file.open("start_centroids.txt");
        file << k << "\n";
        for (int i = 0; i < k; i++){
            for (double j : centroids1[i]){
                file << to_string(j) << " ";}
            file << "\n";}
        for (int i = 0; i < k; i++){
            for (double j : centroids2[i]){
                file << to_string(j) << " ";}
            file << "\n";}
        file << "\n";

        file.close();
        // Выравнивание центров кластеров
        vector<vector<vector<double>>> prepared_clusters;
        while (not equal_matrices(centroids1, create_new_centroids(create_clasters(centroids1, all_elems)))){
            prepared_clusters = create_clasters(centroids1, all_elems);
            centroids1 = create_new_centroids(prepared_clusters);
        };
        cout<<centroids1.size()<<endl;
        centroids4 = iso_data(prepared_clusters, porog, Nmin, dmin);
       
        file.open("iso.txt");
        for (int i = 0; i < centroids4.size(); i++){
            for (double j : centroids4[i]){
                file << to_string(j) << " ";}
            file << "\n";}
        file.close();

        while (not equal_matrices(centroids2, create_new_centroids(create_clasters(centroids2, all_elems)))){
            centroids2 = create_new_centroids(create_clasters(centroids2, all_elems));
        };
        // Вывод итоговых центров кластеров
        file.open("k_means.txt");
        file << k << "\n";
        for (int i = 0; i < k; i++){
            for (double j : centroids1[i]){
                file << to_string(j) << " ";}
            file << "\n";}
        for (int i = 0; i < k; i++){
            for (double j : centroids2[i]){
                file << to_string(j) << " ";}
            file << "\n";}
        file.close();
        return;
    };
};


int main()
{
    ClassParamsRepository rep("data.txt");
    ClassAll_Points pointkeeper;
    //Заполнение хранилища точек
    ofstream file;
    file.open("file_for_checking.txt");
    file.clear();

    for (int i = 0; i < class_k; i++){
        pointkeeper.addClass(points_for_1Class("file_for_checking.txt", rep.repository[i], class_k_elements[i]));
    };
    //Алгоритм Хо-Кашьяпа
    ofstream filee;
    filee.open("kashyap_lines.txt");
    filee.clear();

    for (int i=0;i<class_k-1;i++) for (int j=i+1;j<class_k;j++){
        kash* kash_ex = new kash(pointkeeper.points_repos[i].point_coordinates, pointkeeper.points_repos[j].point_coordinates);
        kash_ex->psevdo_solution();
        delete kash_ex;
    };
    //Метрики
    //metricks metr;
    //metr.evklidova(pointkeeper.points_repos[0].point_coordinates,pointkeeper.points_repos[1].point_coordinates);
    //metr.manhat(pointkeeper.points_repos[0].point_coordinates,pointkeeper.points_repos[1].point_coordinates);

    // Кластеры
    create_clusters clast;
    clast.k_means(pointkeeper, 3);
    return 0;
}