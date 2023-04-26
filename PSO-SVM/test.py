import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('myapp.ui', self)
        self.show()

        # 加载数据集
        self.iris = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.iris.data,
                                                                                self.iris.target,
                                                                                test_size=0.2,
                                                                                random_state=42)
        # 创建逻辑回归模型
        self.model = LogisticRegression()

        # 配置视图元素
        self.comboBox.addItems(['逻辑回归'])

        # 连接事件处理程序
        self.pushButton.clicked.connect(self.browse)
        self.predictButton.clicked.connect(self.predict)

    def browse(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择文件", ".")
        self.filePathTextEdit.setText(filename)

    def predict(self):
        # 获取模型选择
        model_type = self.comboBox.currentText()

        # 根据模型选择进行预测
        if model_type == '逻辑回归':
            # 加载输入数据
            input_file = self.filePathTextEdit.toPlainText()
            try:
                with open(input_file) as f:
                    # 在这里将文件内容转换为模型需要的格式，此处将其转为花瓣长度、花瓣宽度、花萼长度、花萼宽度四个属性
                    data = f.read()
                    input_data = [[float(x) for x in line.split() if x != ''] for line in data.split('\n') if line != '']
                # 训练模型
                self.model.fit(self.X_train, self.y_train)

                # 进行预测
                y_pred = self.model.predict(input_data)

                # 计算准确率和输出
                accuracy = accuracy_score(self.y_test, y_pred)
                self.resultLabel.setText(str(y_pred))
                self.accuracyLabel.setText(str(accuracy))
            except Exception as e:
                QMessageBox.critical(self, "错误", "文件加载错误：%s" % str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    sys.exit(app.exec_())

