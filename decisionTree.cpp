/*
 Description : A C++ implementation of a decision tree classifier based on the
               ID3 (Iterative Dichotomiser 3) algorithm.
               The program reads a dataset from a CSV file,
               builds a predictive model by recursively splitting the data
               based on information gain, and then allows for interactive
               predictions on new, unseen data instances.

 Expected File Format:
 -   Header Row: The first line of the file is the header row,
     containing the names of the features and the target variable.
 -   Delimiter: Values must be separated by commas (,).
 -   Data Type: All data is treated as categorical (string) data.
 -   No Missing Values: The program does not handle missing values.

 Interactive Prediction Format:
 When prompted, enter feature-value pairs separated by commas, like so:
 > feature1=value1,feature2=value2,feature3=value3
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <memory>

struct TreeNode
{
    std::string feature;
    std::string value;
    std::string prediction;
    bool isLeaf;
    std::vector<std::unique_ptr<TreeNode>> children;

    TreeNode() : isLeaf(false) {}
};

class DecisionTree
{
private:
    std::vector<std::vector<std::string>> data;
    std::vector<std::string> headers;
    std::string targetColumn;
    std::unique_ptr<TreeNode> root;

    // Parse CSV file
    bool loadCSV(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return false;
        }

        std::string line;
        bool isFirstLine = true;

        while (std::getline(file, line))
        {
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ','))
            {
                // Remove whitespace
                cell.erase(0, cell.find_first_not_of(" \t"));
                cell.erase(cell.find_last_not_of(" \t") + 1);
                row.push_back(cell);
            }

            if (isFirstLine)
            {
                headers = row;
                isFirstLine = false;
            }
            else
            {
                data.push_back(row);
            }
        }

        file.close();
        return true;
    }

    // Calculate entropy
    double calculateEntropy(const std::vector<int> &indices)
    {
        if (indices.empty())
            return 0.0;

        std::map<std::string, int> counts;
        int targetIdx = getColumnIndex(targetColumn);

        for (int idx : indices)
        {
            counts[data[idx][targetIdx]]++;
        }

        double entropy = 0.0;
        int total = indices.size();

        for (const auto &pair : counts)
        {
            double prob = static_cast<double>(pair.second) / total;
            if (prob > 0)
            {
                entropy -= prob * log2(prob);
            }
        }

        return entropy;
    }

    // Calculate information gain
    double calculateInformationGain(const std::vector<int> &indices, const std::string &feature)
    {
        double parentEntropy = calculateEntropy(indices);
        int featureIdx = getColumnIndex(feature);

        // Group by feature values
        std::map<std::string, std::vector<int>> groups;
        for (int idx : indices)
        {
            groups[data[idx][featureIdx]].push_back(idx);
        }

        double weightedEntropy = 0.0;
        int total = indices.size();

        for (const auto &group : groups)
        {
            double weight = static_cast<double>(group.second.size()) / total;
            weightedEntropy += weight * calculateEntropy(group.second);
        }

        return parentEntropy - weightedEntropy;
    }

    // Get column index by name
    int getColumnIndex(const std::string &columnName)
    {
        auto it = std::find(headers.begin(), headers.end(), columnName);
        return it != headers.end() ? std::distance(headers.begin(), it) : -1;
    }

    // Find best feature to split on
    std::string findBestFeature(const std::vector<int> &indices, const std::set<std::string> &usedFeatures)
    {
        std::string bestFeature;
        double bestGain = -1.0;

        for (const std::string &feature : headers)
        {
            if (feature != targetColumn && usedFeatures.find(feature) == usedFeatures.end())
            {
                double gain = calculateInformationGain(indices, feature);
                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = feature;
                }
            }
        }

        return bestFeature;
    }

    // Get most common class
    std::string getMostCommonClass(const std::vector<int> &indices)
    {
        std::map<std::string, int> counts;
        int targetIdx = getColumnIndex(targetColumn);

        for (int idx : indices)
        {
            counts[data[idx][targetIdx]]++;
        }

        std::string mostCommon;
        int maxCount = 0;
        for (const auto &pair : counts)
        {
            if (pair.second > maxCount)
            {
                maxCount = pair.second;
                mostCommon = pair.first;
            }
        }

        return mostCommon;
    }

    // Check if all instances have same class
    bool allSameClass(const std::vector<int> &indices)
    {
        if (indices.empty())
            return true;

        int targetIdx = getColumnIndex(targetColumn);
        std::string firstClass = data[indices[0]][targetIdx];

        for (int idx : indices)
        {
            if (data[idx][targetIdx] != firstClass)
            {
                return false;
            }
        }

        return true;
    }

    // Build decision tree recursively
    std::unique_ptr<TreeNode> buildTree(const std::vector<int> &indices, std::set<std::string> usedFeatures)
    {
        auto node = std::make_unique<TreeNode>();

        // Base cases
        if (indices.empty())
        {
            node->isLeaf = true;
            node->prediction = "Unknown";
            return node;
        }

        if (allSameClass(indices))
        {
            node->isLeaf = true;
            node->prediction = data[indices[0]][getColumnIndex(targetColumn)];
            return node;
        }

        // Find best feature
        std::string bestFeature = findBestFeature(indices, usedFeatures);
        if (bestFeature.empty())
        {
            node->isLeaf = true;
            node->prediction = getMostCommonClass(indices);
            return node;
        }

        node->feature = bestFeature;
        usedFeatures.insert(bestFeature);
        int featureIdx = getColumnIndex(bestFeature);

        // Group by feature values
        std::map<std::string, std::vector<int>> groups;
        for (int idx : indices)
        {
            groups[data[idx][featureIdx]].push_back(idx);
        }

        // Create children
        for (const auto &group : groups)
        {
            auto child = buildTree(group.second, usedFeatures);
            child->value = group.first;
            node->children.push_back(std::move(child));
        }

        return node;
    }

    // Print tree recursively
    void printTree(const TreeNode *node, int depth = 0, const std::string &parentValue = "")
    {
        if (!node)
            return;

        std::string indent(depth * 2, ' ');

        if (node->isLeaf)
        {
            std::cout << indent << "-> " << node->prediction << std::endl;
        }
        else
        {
            if (depth > 0)
            {
                std::cout << indent << "if " << node->feature << " == " << parentValue << ":" << std::endl;
            }
            else
            {
                std::cout << "Root: " << node->feature << std::endl;
            }

            for (const auto &child : node->children)
            {
                if (!node->feature.empty())
                {
                    std::cout << indent << "  " << node->feature << " == " << child->value << ":" << std::endl;
                }
                printTree(child.get(), depth + 1, child->value);
            }
        }
    }

    // Predict using the tree
    std::string predict(const TreeNode *node, const std::map<std::string, std::string> &instance)
    {
        if (!node)
            return "Unknown";

        if (node->isLeaf)
        {
            return node->prediction;
        }

        auto it = instance.find(node->feature);
        if (it == instance.end())
        {
            return "Unknown";
        }

        std::string featureValue = it->second;

        for (const auto &child : node->children)
        {
            if (child->value == featureValue)
            {
                return predict(child.get(), instance);
            }
        }

        return "Unknown";
    }

public:
    bool train(const std::string &filename, const std::string &target)
    {
        targetColumn = target;

        if (!loadCSV(filename))
        {
            return false;
        }

        if (data.empty())
        {
            std::cerr << "Error: No data loaded" << std::endl;
            return false;
        }

        if (getColumnIndex(targetColumn) == -1)
        {
            std::cerr << "Error: Target column '" << targetColumn << "' not found" << std::endl;
            return false;
        }

        // Create indices for all data
        std::vector<int> allIndices;
        for (int i = 0; i < data.size(); i++)
        {
            allIndices.push_back(i);
        }

        std::set<std::string> usedFeatures;
        root = buildTree(allIndices, usedFeatures);

        return true;
    }

    void printDecisionTree()
    {
        if (root)
        {
            std::cout << "\nDecision Tree Structure:" << std::endl;
            std::cout << "========================" << std::endl;
            printTree(root.get());
        }
        else
        {
            std::cout << "No tree built yet." << std::endl;
        }
    }

    std::string predictInstance(const std::map<std::string, std::string> &instance)
    {
        return predict(root.get(), instance);
    }

    void printDataInfo()
    {
        std::cout << "Dataset Information:" << std::endl;
        std::cout << "===================" << std::endl;
        std::cout << "Rows: " << data.size() << std::endl;
        std::cout << "Columns: " << headers.size() << std::endl;
        std::cout << "Features: ";
        for (const std::string &header : headers)
        {
            std::cout << header << " ";
        }
        std::cout << std::endl;
        std::cout << "Target: " << targetColumn << std::endl
                  << std::endl;
    }
};

int main()
{
    DecisionTree tree;
    std::string filename, targetColumn;

    std::cout << "Decision Tree Builder" << std::endl;
    std::cout << "====================" << std::endl;

    std::cout << "Enter CSV filename: ";
    std::getline(std::cin, filename);

    std::cout << "Enter target column name: ";
    std::getline(std::cin, targetColumn);

    if (tree.train(filename, targetColumn))
    {
        tree.printDataInfo();
        tree.printDecisionTree();

        // Interactive prediction
        std::cout << "\nInteractive Prediction Mode" << std::endl;
        std::cout << "===========================" << std::endl;
        std::cout << "Enter 'quit' to exit" << std::endl;

        std::string input;
        while (true)
        {
            std::cout << "\nEnter feature values (format: feature1=value1,feature2=value2): ";
            std::getline(std::cin, input);

            if (input == "quit")
                break;

            std::map<std::string, std::string> instance;
            std::stringstream ss(input);
            std::string pair;

            while (std::getline(ss, pair, ','))
            {
                size_t pos = pair.find('=');
                if (pos != std::string::npos)
                {
                    std::string feature = pair.substr(0, pos);
                    std::string value = pair.substr(pos + 1);

                    // Remove whitespace
                    feature.erase(0, feature.find_first_not_of(" \t"));
                    feature.erase(feature.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);

                    instance[feature] = value;
                }
            }

            if (!instance.empty())
            {
                std::string prediction = tree.predictInstance(instance);
                std::cout << "Prediction: " << prediction << std::endl;
            }
            else
            {
                std::cout << "Invalid input format. Use: feature1=value1,feature2=value2" << std::endl;
            }
        }
    }

    return 0;
}