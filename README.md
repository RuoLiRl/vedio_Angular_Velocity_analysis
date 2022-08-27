# vedio_Angular_Velocity_analysis
 input:video output:ω-t graph(and a linear model)
输入视频，输出角速度-时间图像（还附带线性拟合功能


操作：
开始时选一个图片作为控制台(consol)图片
共3个操作
getRect:选取边界；fine:调整边界/图像处理参数；output:进行直线拟合并输出结果

getRect选择边界点后按任意键退出

fine过程中可以直接ctrl+c或q退出并查看结果
fine/getRect过程中按s可以调试(更改图像处理方面的参数)，按q即可退出

output后：
选中consol窗口
按a显示所有点
按n显示拟合曲线
按s显示拟合曲线所用点
按c清除所有点
按0-7输出帧数跨度为2-9的点，9输出跨度11的点，10输出跨度13的点（可能数字不准确，但反正是不同点