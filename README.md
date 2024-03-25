# 索力调整遗传算法 / Tension Adjustment Genetic Algorithm

本项目是一个基于遗传算法的索力调整工具，旨在为桥梁等土木工程结构提供一个自动化、优化的索力调整方案。项目采用Python编程语言开发，提供了图形用户界面，使得操作直观简单。

This project is a tension adjustment tool based on genetic algorithms, designed to provide an automated and optimized tension adjustment solution for civil engineering structures such as bridges. It is developed in Python and offers a graphical user interface for intuitive operation.

## 功能特点 / Features

- **智能优化**：利用遗传算法自动寻找最优或近似最优的索力调整方案。
- **用户友好**：提供图形用户界面，易于输入参数和解读结果。
- **数据可视化**：展示算法进度和结果，包括索力调整前后的对比等。
- **灵活适用**：适用于不同类型和规模的桥梁工程项目。

- **Intelligent Optimization**: Automatically finds the best or near-optimal tension adjustment solutions using genetic algorithms.
- **User-Friendly**: Offers a graphical user interface, making it easy to input parameters and interpret results.
- **Data Visualization**: Displays the progress and results of the algorithm, including comparisons before and after tension adjustment.
- **Versatile Application**: Suitable for bridge engineering projects of different types and sizes.

## 安装指南 / Installation Guide

### 环境要求 / Environment Requirements

- Python 3.6 或更高版本
- Python 3.6 or higher version

### 安装步骤 / Installation Steps

1. 克隆仓库到本地 / Clone the repository to local:
git clone https://github.com/zjkl19/GeneticAlgorithmForBridgeTension.git
2. 切换到项目目录 / Switch to the project directory:
cd GeneticAlgorithmForBridgeTension
3. 创建并激活虚拟环境 / Create and activate a virtual environment:
- Windows:
  ```
  python -m venv venv
  .\venv\Scripts\activate
  ```
- MacOS/Linux:
  ```
  python3 -m venv venv
  source venv/bin/activate
  ```
4. 安装依赖 / Install dependencies:
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
## 使用说明 / Usage

启动应用程序，请在命令行中运行以下命令：

To start the application, run the following command in the terminal:

python main.py

按照图形界面的指示输入相关参数，包括初始索力、目标索力和影响矩阵等，然后点击“开始计算”按钮。计算完成后，结果将在界面上显示。

Follow the instructions on the graphical interface to input relevant parameters, including initial tension, target tension, and the influence matrix, then click the "Start Calculation" button. The results will be displayed on the interface upon completion.

## 开发背景 / Background

在土木工程领域，特别是桥梁建设和维护中，索力调整是确保结构安全和稳定的关键步骤。传统的索力调整方法通常依赖于经验和试错，效率低下且难以达到最优解。通过引入遗传算法，我们旨在提供一种智能化、高效的索力调整方案，以优化桥梁的结构表现和延长其使用寿命。

In the field of civil engineering, especially in the construction and maintenance of bridges, tension adjustment is a key step to ensure structural safety and stability. Traditional tension adjustment methods often rely on experience and trial and error, which are inefficient and difficult to achieve optimal solutions. By introducing genetic algorithms, we aim to provide an intelligent and efficient tension adjustment solution to optimize the structural performance of bridges and extend their service life.

## 贡献指南 / Contribution Guide

我们欢迎并鼓励社区成员对此项目作出贡献，无论是通过报告问题、提出改进建议，还是直接参与代码贡献。如果您有任何想法或建议，请通过Issue或Pull Request与我们分享。

We welcome and encourage community members to contribute to this project, whether it's by reporting issues, suggesting improvements, or directly contributing to the code. If you have any ideas or suggestions, please share them with us through Issues or Pull Requests.

## 许可证 / License

本项目采用MIT许可证，详情请见LICENSE文件。

This project is licensed under the MIT License - see the LICENSE file for details.

## 联系方式 / Contact

如有任何问题或建议，请通过以下方式联系我们：

If you have any questions or suggestions, please contact us through the following:

- Email: 待定

