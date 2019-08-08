day-day-up
============
记录每天的学习
所有记录暂时移植到readme
从11月开始链接附上 尽量做到详细


9月17日
1 动态规划 0-1背包问题。
2 cnn特征图大小计算。
3 线性回归的描述以及基本假设。
4 leetcode 无重复字符的最长子串，复杂度O(n2)会超时（使用字典滑动窗口）
5 svm线性可分部分

9月18日
1 matrix = [[False for i in range(6)] for j in range(6)]
2 leetcode 动态规划求解最长回文子串
3 hmm 前向计算法,结合hanlp博客与李航的统计学一起看的

9月19日
1 句法依存分析以及语法树
2 leetcode双指针盛最多水问题
3 hmm 维特比算法
4 jupyter notebook 写文档

9月20日
1 leetcode 二分查找
2 二叉树取第k小的数（中序遍历或者使用栈迭代）
3 熵，交叉熵，kL
4 GMM混合高斯模型


9月21-9月24日 中秋

9月25日
1 leetcode最长公共前缀
2 生成模型、朴素贝叶斯、线性模型、核模型
3 正则化L1,L2

9月26日
1 leetcode矩阵90度翻转 zip(*m[::-1])
2 u-net,fcn模型以及upsample，反卷积

9月28日
1 二分查找、选择排序
2 递归
3 欧几里得算法求最大公约数
4 快速排序
5 散列表
6 cnn边缘特征提取，抵消。

9月29日-10月7日 国庆以及请假两天

10月8日
1 欧几里德算法复习
2 广度优先搜索与深度优先搜索
3 dj搜索算法
4 svm核函数
5 adaboost算法

10月9日
1 leetcode全排列回溯
2 lda基本概念

10月10日
1 分位数 pd.qutile
2 对数变化处理具有重尾分布的重要工具，对数变化后，直方图不集中在低端，而变得更分散在x轴
3 动态规划算法填表，每个子问题都是离散的，即不依赖于其他子问题，动态规划才管用
4 贪心算法
5 背包问题、最长公共子串，最长公共子序列，最长公共子序列问题的解并不一定在最后一格。

10月11日
1 leetcode 强盗相邻房间抢劫问题
2 样本标准差与样本标准误差
3 随机变量、大数定律、样本抽样
4 中心极限定理（无限接近正态分布）

10月12日
1 leetcode装水最多问题
2 tf-idf算法
3 pca算法

10月13日
1 时间序列预处理
2 纯随机序列，白噪声序列
3 平稳性检验（时序图检验、自相关图检验、单位根检验）
4 截尾、拖尾
5 leetcode minumum path sum

10月14日
1 leetcode 回溯找出不重复元素subset
2 时间序列模型代码

10月15日
1 leetcode 三数之和最接近
2 np.random.permutation() 随机打乱
3 tf.feature_column.numeric_column()

10月17日
1 zip与zip(*)压缩与解压的关系
2 squeeze()维度变为1
3 time.strftime()
4 np.newaxis()列移到行上 新增加一列
5 lstm股票预测 同步预测与异步预测

10月18日
1 '&' 8 & 9 = 8
2 leetcode 二进制1的个数 d[p] = d[p&(p-1)] + 1
3 拉格朗日乘子法
4 lda 线性判别算法
5 tr（矩阵对角线元素之和，特征值之和）

10月19日
1 leetcode
2 fft 变化 图像特征提取

10月22日
1 罗马数字转化为整数
2 fft资料

10月23日
1 leetcode Letter Combinations of a phone number (递归)
2 递增式的学习 partial_fit
3 决策树、knn

10月24日
1 递归题目 ‘ort’原则 选择 限制 结束条件
2 leetcode generate parentheses
3 kd树切分坐标轴 l = j%k + 1
4 kmeans 异常检测

10月25日
1 堆排序
2 用电量异常预测
3 价值客户分类
4 指标计算，构建特征

10月26日
1 leetcode swap nodes in paris
10月29日
1 defaultdict key 不存在 返回默认值
2 Counter 简单计数器
3 Counter(B).items()
4 tensorflow cnn

10月30日
1 python ord chr
2 rnn，lstm

10月31日
1 leetcode 周長4 重叠-2
2 遗传算法

11月1日
1 leetcode 爬楼梯问题
2 dssm （Deep Structured Senmantic Models）

# 11月2日
以后将附上所阅资料
>
1 logistic回归含义后验概率分布，用概率的角度去设计类然函数，logistic模型相比于感知机模型对异常数据具有更好的鲁棒性
>
###
[线性分类模型（二）：logistic回归模型分析](https://mp.weixin.qq.com/s?__biz=MzU0MDQ1NjAzNg==&mid=100000600&idx=1&sn=a94872004daaafadcba76600893e35ef&chksm=7b39a6534c4e2f4551b800f3f27115c9d95fb612c107c589ec0436aedd85ecfa9b630bd9f295%23rd)
>
2 leetcode 121 Best Time to Buy and Sell Stock
>
第一种 时间超过
>
### 对于从第一天后的每一天i来说：

    如果我们在第i天卖出，则能赚的钱是在第i-1卖能赚到的钱+（第i天的股价 - 第i-1的股价）
    如果我们在第i天不卖出，则当前赚的钱为 0
>
###

    class Solution:
        def maxProfit(self, prices):
            """
            :type prices: List[int]
            :rtype: int
            """
            p = 0
            for i in range(len(prices)):
                for j in range(i,len(prices)):
                    if prices[j] - prices[i] > p:
                        p = prices[j] - prices[i]
            return p

      def maxProfit(self, prices):
          """
          :type prices: List[int]
          :rtype: int
          """
          if not prices or len(prices) == 0:
              return 0
          opt = [0] * len(prices)
          for i in range(1, len(prices)):
              opt[i] = max(opt[i-1]+prices[i]-prices[i-1], 0)
          return max(opt)

      def maxProfit(self, prices):
          """
          :type prices: List[int]
          :rtype: int
          """
          if not prices or len(prices) == 0:
              return 0
          res, max_cur = 0, 0
          for i in range(1, len(prices)):
              max_cur = max(0, max_cur+prices[i]-prices[i-1]) 
              res = max(res, max_cur)
          return res
 
 # 11月5日
 1 leetcode Min Cost Climbing Stairs
>
###
    class Solution:
        def minCostClimbingStairs(self, cost):
            """
            :type cost: List[int]
            :rtype: int
            """
            dp = [0] * (len(cost) + 1)
            for i in range(2,len(cost) + 1):
                dp[i] = min(dp[i - 2] + cost[i - 2],dp[i - 1] + cost[i - 1])
            return dp[len(cost)]

# 11月6日
1 leetcode 最大连续子串和
>
###
    class Solution:
        def maxSubArray(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            dp = [0] * len(nums)
            for i in range(len(nums)):
                dp[i] = max(dp[i - 1] + nums[i],nums[i])
            return max(dp)
>
2 gensim word2vector
>
###
[word2vec的应用----使用gensim来训练模型](https://blog.csdn.net/qq_35273499/article/details/79098689)
>
3 汉子编码区间
>
###
https://blog.csdn.net/m372897500/article/details/37592543
>
4 python 生成器以及迭代器
>
###
https://blog.csdn.net/on_1y/article/details/8640012#sec-10
>
5 embedding层理解以及如何使用keras加载预训练词向量
>
###
https://blog.csdn.net/jiangpeng59/article/details/77533309
>
6 期望、方差、协方差
>
###
https://blog.csdn.net/MissXy_/article/details/80705828

# 11月7日
1 模型复杂度
>
###
[深入理解线性回归算法（二）：正则项的详细分析](https://mp.weixin.qq.com/s?__biz=MzU0MDQ1NjAzNg==&mid=2247484149&idx=1&sn=c0b0a425eb081e125407d6cec418b144&chksm=fb39a7fecc4e2ee871972a8c54454172affa58dfb01b108c1c93983be4dffe0df9112479e612&token=1082697960&lang=zh_CN&scene=21#wechat_redirect)
>
2 感知机 
>
和李航的有点区别（损失函数构造不一样）
>
###
[python感知机](https://github.com/basicv8vc/Python-Machine-Learning-zh/blob/master/第二章/机器学习分类算法.ipynb)

# 11月8日
1.  Trim a Binary Search Tree(669)
>
###
    这题做法有些取巧，并不是真正意义上在内存里面删除不符合区间的Node，只是将Node的指向进行的更改，大致思路：

    每一层的Condition有三种：

    root.val小于区间的lower bound L，则返回root.right subtree传上来的root，这里就变相的'删除'掉了当前root和所有root.left的node
    root.val大于区间的upper bound R，则返回root.left subtree传上来的root
    满足区间，则继续递归
    当递归走到叶子节点的时候，我们向上返回root，这里return root的定义是：
    返回给parent一个区间调整完以后的subtree

###
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None

    class Solution(object):
        def trimBST(self, root, L, R):
            # 每一层的Condition
            if root == None:
                return root
            if root.val > R:
                return self.trimBST(root.left,L,R)
            if root.val < L:
                return self.trimBST(root.right,L,R)
            # 再区间内，正常的Recursion
            root.right = self.trimBST(root.right,L,R)
            root.left = self.trimBST(root.left,L,R)

            # 返回给parent一个区间调整完以后的subtree
            return root
 >
 2 python cumsum
 >
 ###
 https://blog.csdn.net/banana1006034246/article/details/78841461
 > 
 3 np.around 返回四舍五入后的值，可指定精度。
 >
 4 时间序列基本模型

# 11月9日
>
1 Distribute Candies(hashtable)
>
2 Flipping an Image
>adf 检验数据平稳
###
https://blog.csdn.net/weixin_42382211/article/details/81332431

# 11月10日
>

###
    1 Merge Two Binary Trees
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None

    class Solution:
        def mergeTrees(self, t1, t2):
            """
            :type t1: TreeNode
            :type t2: TreeNode
            :rtype: TreeNode
            """
            if not t1 and not t2:
                return None
            if not t1 or not t2:
                return t1 or t2
            node = TreeNode(t1.val+t2.val)
            node.left = self.mergeTrees(t1.left, t2.left)
            node.right = self.mergeTrees(t1.right, t2.right)
            return node 
>
# 11月13日
1  Average of Levels in Binary Tree（637）
###
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None

    class Solution:
        def averageOfLevels(self, root):
            averages = []
            level = [root]
            while level:
                averages.append(sum(node.val for node in level) / len(level))
                level = [kid for node in level for kid in (node.left, node.right) if kid]
            return averages
>
2 np.full((2, 2), 10)
>
###
    array([[10, 10],
           [10, 10]])
>
3 pandas 删除数值的情况
###
    https://www.cnblogs.com/cocowool/p/8421997.html

>
# 11月14日
1 Uncommon Words from Two Sentences 
>
###
    class Solution:
    def uncommonFromSentences(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: List[str]
        """
        count = {}
        for word in A.split():
            count[word] = count.get(word, 0) + 1
        for word in B.split():
            count[word] = count.get(word, 0) + 1

        #Alternatively:
        #count = collections.Counter(A.split())
        #count += collections.Counter(B.split())

        return [word for word in count if count[word] == 1]
2 python 集合处理
>
###
https://blog.csdn.net/Chihwei_Hsu/article/details/81416818
    
>
3 刀具磨损预测baseline model
>
4 使用sklearn库中的SVR做回归分析
###
    http://www.dataivy.cn/blog/regression_with_sklearn/

# 11月15日
>
1 Array Nesting（565）
###
    class Solution:
    def arrayNesting(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 1
        visited = set()
        for i,num in enumerate(nums):
            length = 1
            start, nxt = num, nums[num]
            if num in visited:
                continue
            while start != nxt:
                length += 1
                nxt = nums[nxt]
                visited.add(nxt)
            res = max(res, length)
        return res
>
2 混淆矩阵
>
![github](https://github.com/Juary88/day-day-up/blob/master/pic/confusion.png)
>
3 似然函数经典理解
###
    1 设总体的概率模型为F(x|θ)。为了说明的方便，暂假定只有一个未知参数，X1，X2，……，Xn是容量为 n 的随机样本（大写X），实际观测到的样本观测值（小写x）为 Xl=x1，X2=x2，……，Xn=xn 。把同各Xi对应的密度函数或概率函数(包括作为未知数的未知参数)的连乘积看成是未知参数的函数，称其为似然函数(Likelihood function)。

    2 为什么需要取log? 
        那么为什么在最优化的时候需要取log呢？有两点原因。 
        1.为了求解简单，在求导的时候。 
        2.为了避免数值的下溢。因为L(θ|x)是由很多概率相乘，而每个概率都是小于一的，如果样本量很大的时候，那么很容易导致L(θ|x)非常非常的小。
###
https://www.cnblogs.com/zhsuiy/p/4822020.html
>
https://blog.csdn.net/lwq1026/article/details/70161857
>
4. logistic回归损失函数 可利用最大似然估计理解 y = 0与y = 1的情况统一公示乘起来
>
5. 多分类softmax

# 11月16日
1 Sort Array By Parity II
>
###
    class Solution:
    def sortArrayByParityII(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        res = [0] * len(A)
        odd = [i for i in A if i % 2 == 0]
        even = [i for i in A if i % 2 == 1]
        for i in range(len(A)):
            if i % 2 == 0:
                res[i] = odd.pop()
            else:
                res[i] = even.pop()
        return res
>
2 感知机中损失函数1/||w||为什么可以不考虑（或直接忽略）？
>
###
    1、1/||w||不影响-y(w,x+b)正负的判断，即不影响学习算法的中间过程。因为感知机学习算法是误分类驱动的，这里需要注意的是所谓的“误分类驱动”指的是我们只需要判断-y(wx+b）的正负来判断分类的正确与否，而1/||w||并不影响正负值的判断。所以1/||w||对感知机学习算法的中间过程可有可无；
    2、1/||w||不影响感知机学习算法的最终结果。因为感知机学习算法最终的终止条件是所有的输入都被正确分类，即不存在误分类的点。则此时损失函数为0. 对应于-y（wx+b）/||w||，即分子为0.则可以看出1/||w||对最终结果也无影响。
    3、对应于svm中的函数间隔。
    4、所以感知机的结果不是唯一的，只考虑分类的正确与否。

# 11月17日
>
1 Number Complement
>
###
    class Solution:
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        s = bin(num)[2:]
        res = 0
        for i in range(len(s) - 1,-1,-1):
            if s[i] == '0':
                res += pow(2,len(s) - 1 - i)
        return res
 >
 2 iou的理解
 >
 3 padding（same 与 valid）
 >
 ###
      If padding == "SAME":
          output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])

        If padding == "VALID":
          output_spatial_shape[i] =
            ceil((input_spatial_shape[i] -
                  (spatial_filter_shape[i]-1) * dilation_rate[i])
                 / strides[i]).

      Raises:
        ValueError: If input/output depth does not match filter shape, if padding
          is other than "VALID" or "SAME", or if data_format is invalid.
>
###
https://oldpan.me/archives/tf-keras-padding-vaild-same
>
4 yolo v1
>
###
https://blog.csdn.net/app_12062011/article/details/77554288

# 11月19日
>
1 leetcode recentcount
>
###
    class RecentCounter:

    def __init__(self):
        self.que = collections.deque()

    def ping(self, t):
        """
        :type t: int
        :rtype: int
        """
        while self.que and self.que[0] < t - 3000:
            self.que.popleft()
        self.que.append(t)
        return self.que
>
### collection
>
https://blog.csdn.net/windanchaos/article/details/77019008

2  yolov1
>
###
https://blog.csdn.net/app_12062011/article/details/77554288
>
https://www.jianshu.com/p/13ec2aa50c12

# 11月20日
1 AI资料
>
[AI算法工程师](http://www.huaxiaozhuan.com/?from=groupmessage&isappinstalled=0)
>
2 自然语言处理好的blog
>
[blog](https://blog.csdn.net/guotong1988/article/category/6076360/2)

3 Shortest Distance to a Character
>
###
    class Solution:
    def shortestToChar(self, S, C):
        """
        :type S: str
        :type C: str
        :rtype: List[int]
        """
        a = []
        ans = []
        for i in range(len(S)):
            if S[i] == C:
                a.append(i)
        for i in range(len(S)):
            t = float('inf')
            for j in a:
                if t > abs(i - j):
                    t = abs(i - j)
            ans.append(t)
        return ans

# 11月21日
>
1 [LeetCode] 647. Palindromic Substrings 回文子字符串
###
    def countSubstrings(self, s):
    if not s:
        return 0

    n = len(s)
    table = [[False for x in range(n)] for y in range(n)]
    count = 0

    # Every isolated char is a palindrome
    for i in range(n):
        table[i][i] = True
        count += 1

    # Check for a window of size 2
    for i in range(n-1):
        if s[i] == s[i+1]:
            table[i][i+1] = True
            count += 1

    # Check windows of size 3 and more
    for k in range(3, n+1):
        for i in range(n-k+1):
            j = i+k-1
            if table[i+1][j-1] and s[i] == s[j]:
                table[i][j] = True
                count += 1

    return count
 
 >
 2 fft转化为频谱
 >
 ###
 https://plot.ly/matplotlib/fft/
 
 # 11月22日
 > 
1 leetcode Arithmetic Slices
>
###
    class Solution:
        def numberOfArithmeticSlices(self, A):
            """
            :type A: List[int]
            :rtype: int
            """
            dp = [0] * len(A)
            for i in range(2,len(A)):
                if A[i] - A[i - 1] == A[i - 1] - A[i - 2]:
                    dp[i] = 1 + dp[i - 1]
            return sum(dp)

# 11月23日
>
1 Is Subsequence
###
    class Solution:
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) == 0:
            return True
        if len(t) == 0:
            return False
        n = 0
        for i in t:
            if s[n] == i:
                n += 1
            if n == len(s):
                return True
        return False

# 11月26日
>
1 sell stock III
>
###
https://github.com/apachecn/awesome-algorithm/blob/master/docs/Leetcode_Solutions/Python/188._Best_Time_to_Buy_and_Sell_Stock_IV.md
>
2 期望泛化误差 = 方差 + 偏差 + 噪声
>

# 11月27日
>
1 python 机器学习实战：信用卡欺诈异常值检测
###
[python 机器学习实战：信用卡欺诈异常值检测](https://blog.csdn.net/dengheCSDN/article/details/79079364#commentBox)
>

# 11月28日
>
1 numpy 实现 fft变化
###
[python傅立叶变化]（http://bigsec.net/b52/scipydoc/frequency_process.html#hilbert）
https://blog.csdn.net/ouening/article/details/71079535
>
[功率谱](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html)

# 11月30日
###
1 [论文在线阅读](https://max.book118.com/html/2017/0704/120163690.shtm)

# 12月5日
###
1 [中文文本中如何抽取关系/观点词](https://www.zhihu.com/question/44633151)
>
2 [nlp任务项目](https://zhuanlan.zhihu.com/p/51279338)
>
3 [nlp专栏](https://www.zhihu.com/people/cangshengtage/activities)
>
  https://www.zhihu.com/topic/19560026/hot
>
4 [face++人脸识别](https://github.com/zhouxuanxian/FaceCompare/blob/master/Face%2B%2BDetectCompare.py)

# 12月6日
1 [对phm数据进行分析](https://blog.csdn.net/qq_40587575/article/details/83351960)
>

# 12月8日
1 [python基础](https://study.163.com/course/introduction.htm?courseId=1005461017)
>

# 12月10日
1 hmm 前向算法理解
![github](https://github.com/Juary88/day-day-up/blob/master/pic/hmm.png)

# 12月11日
1 [使用Python中的Featuretools实现自动化特征工程的实用指南](https://www.ziiai.com/blog/772)

# 12月14日
1 leetcode 841. Keys and Rooms
### 
    class Solution:
        def canVisitAllRooms(self, rooms):
            """
            :type rooms: List[List[int]]
            :rtype: bool
            """
            visited = [0] * len(rooms)
            self.dfs(rooms,0,visited)
            return sum(visited) ==  len(rooms)

        def dfs(self,rooms,index,visited):
            visited[index] = 1
            for key in rooms[index]:
                if visited[key] == 0:
                    self.dfs(rooms,key,visited)

2 [Inception](https://www.cnblogs.com/haiyang21/category/963117.html)

# 12月15日
1 leetcode 695. Max Area of Island
>
###
    class Solution:
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid or len(grid) == 0:
            return 0
        m = len(grid)
        n = len(grid[0]) if m else 0
        
        def dfs(i, j, area):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
                return area
            else:
                grid[i][j] = 0
                area += 1
            for x, y in ([(i, j+1), (i, j-1), (i+1, j), (i-1, j)]):
                area = dfs(x, y, area)
            return area 

        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                area = dfs(i, j, 0)
                res = max(res, area)
        return res   
>
2 [深入理解Batch Normalization批标准化](https://www.cnblogs.com/guoyaohua/p/8724433.html)
>
3 [inceptionv1，v2...](https://blog.csdn.net/qq_14845119/article/details/73648100)
>

# 12月17日
1 leetcode Single Number III(260)
###
    位运算，终于要take it了

    非常常见的一道算法题，将所有数字进行异或操作即可。对于异或操作明确以下三点：

    一个整数与自己异或的结果是0
    一个整数与0异或的结果是自己
    异或操作满足交换律，即a^b=b^a
>
2 [正则验证](https://regexr.com)

# 12月18日
1 leetcode 287. Find the Duplicate Number
>
###
    class Solution:
        def findDuplicate(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            nums = sorted(nums)
            i,j = 0,1
            while nums[i] != nums[j]:
                i += 1
                j += 1
            return nums[i]
 >
 2 [svm kernel](https://blog.csdn.net/aliceyangxi1987/article/details/80617649)
 
 # 12月20日
 1 leetcode 
 ###
    class Solution(object):
    def isNStraightHand(self, hand, W):
        count = collections.Counter(hand)
        while count:
            m = min(count)
            for k in range(m, m+W):
                v = count[k]
                if not v: return False
                if v == 1:
                    del count[k]
                else:
                    count[k] = v - 1

        return True

# 12月22日
1 [pytorch 学习网站]（https://pytorch.apachecn.org/#/）
>
2 [nlp网站]（https://github.com/apachecn/AiLearning）
>
3 [pytorch tutorial](https://github.com/MorvanZhou/PyTorch-Tutorial

# 12月24日
1 [keras文档](https://keras.io/zh/callbacks/)
>
2 [TimeDistributed](https://blog.csdn.net/oQiCheng1234567/article/details/73051251)

# 12月25日
1 [lstm + cnn TimeDistributed](https://blog.csdn.net/luoganttcc/article/details/83582318)
2 [keras lstm详解](https://blog.csdn.net/jiangpeng59/article/details/77646186)
3 [keras 文本分类法律相关demo](https://blog.csdn.net/vivian_ll/article/details/80829474)

# 12月27日
1 [face yolov3](https://juejin.im/post/5b3d943ef265da0fa332cd66)
>
2 [DL-Project-Template](https://github.com/SpikeKing/DL-Project-Template)
>
3 [labelimg](https://blog.csdn.net/yeluohanchan/article/details/79023863)

# 12月28日
1 [yolov3训练自己数据](https://blog.csdn.net/yeluohanchan/article/details/79023863)

# 1月3号
1 [yolo-v3 CarND-Vehicle-Detection](https://github.com/xslittlegrass/CarND-Vehicle-Detection)

# 1月4日
1 leetcode 621. Task Scheduler
###
    class Solution:
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        d = collections.Counter(tasks)
        part_count = max(d.values()) - 1
        empty_slots = part_count * (n-(len([i for i in d.values() if i == max(d.values())])-1))
        available_tasks = len(tasks) - len([i for i in d.values() if i == max(d.values())]) * max(d.values())
        idles = max(0, empty_slots - available_tasks)
        return len(tasks) + idles
# 1月7日
1 leetcode 547. Friend Circles
###
    class Solution(object):
    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        _set=set()
        
        def dfs(node):
            for i,val in enumerate(M[node]):
                if val and i not in _set:
                    _set.add(i)
                    dfs(i)
        result=0
        for i in range(len(M)):
            if i not in _set:
                result+=1
                dfs(i)
        return result
>
2 [yolov3训练自己模型](https://blog.csdn.net/u012746060/article/details/81183006)
>
3 [deeplearning](https://github.com/sea-boat/DeepLearning-Lab)
>
4 [语义分割model](https://github.com/mrgloom/awesome-semantic-segmentation)

# 1月8日
1 [语义分割fcn](https://zhuanlan.zhihu.com/p/31428783)
>
2 keras concatenate
### 
    import numpy as np
    import keras.backend as K
    import tensorflow as tf

    t1 = K.variable(np.array([[[[1, 2], [2, 3]], [[4, 4], [5, 3]]]]))
    t2 = K.variable(np.array([[[[7, 4], [8, 4]], [[2, 10], [15, 11]]]]))
    d0 = K.concatenate([t1 , t2] , axis=0)
    print(d0.shape)
    d1 = K.concatenate([t1 , t2] , axis=1)
    print(d1.shape)
    d2 = K.concatenate([t1 , t2] , axis=2)
    print(d2.shape)
    d3 = K.concatenate([t1 , t2] , axis=3)
    print(d3.shape)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(d0))
        print(sess.run(d1))
        print(sess.run(d2))
>

# 1月9日
1 [归一化方法的两个问题讨论](https://mp.weixin.qq.com/s?__biz=MzU0MDQ1NjAzNg==&mid=2247485585&idx=1&sn=3e9fdbbc9b12644cfb7ab1a84460a078&chksm=fb39ad9acc4e248c9a97045e504553e18fd8a241676fd6f33a332b55c3c40390fa8e7cb9b745&mpshare=1&scene=23&srcid=0109y0c1Zwx2lzrg3vrTlX5o)

>
2 [吴恩达deeplearning.ai](https://github.com/bighuang624/Andrew-Ng-Deep-Learning-notes/tree/master/docs)
>
3 [knife 论文](https://max.book118.com/html/2017/0704/120163690.shtm)
>
4 [u-net demo](https://github.com/mattzheng/U-Net-Demo)
>
5 [人脸识别网站](https://github.com/krasserm/face-recognition)

# 1月11日
1 [Deep face recognition with Keras, Dlib and OpenCV](https://krasserm.github.io/2018/02/07/deep-face-recognition/)
>
2 [深度学习模型 MTCNN 和人脸识别的深度学习模型 FaceNet](https://www.infoq.cn/article/how-to-achieve-face-recognition-using-mtcnn-and-facenet)

# 1月14日
1 [机器学习与自然语言专栏nlp](https://zhuanlan.zhihu.com/qinlibo-ml)
>
2 [逻辑回归与最大似然函数](https://blog.csdn.net/star_liux/article/details/39666737）

# 1月15日
1 [深度学习模型](https://github.com/liuzhuang13/DenseNet)
>
2 [densenet理解](https://blog.csdn.net/u014380165/article/details/75142664)
>
3 
###
    Factorizing Convolutions with Large Filter Size，也就是分解大的卷积，用小的卷积核替换大的卷积核，因为大尺寸的卷积核可以带来更大的感受野，但也意味着更多的参数，比如5x5卷积核参数是3x3卷积核的25/9=2.78倍。因此可以用2个连续的3x3卷积层(stride=1)组成的小网络来代替单个的5x5卷积层，(保持感受野范围的同时又减少了参数量),也就产生了Inception V2;而nxn的卷积核又可以通过1xn卷积后接nx1卷积来替代,也就是Inception V3结构,但是作者发现在网络的前期使用这种分解效果并不好，还有在中度大小的feature map上使用效果才会更好。（对于mxm大小的feature map,建议m在12到20之间）
>
4 [hog + svm目标检测](https://zhuanlan.zhihu.com/p/34818568)
>
5 [自然语言处理文本分类三种对比方法](https://zhuanlan.zhihu.com/p/34688488)
>
6 [QA Model](https://github.com/l11x0m7/Question_Answering_Models)
>

# 1月17日
1 [信号处理](https://zhuanlan.zhihu.com/xinhao)
>
2 [经验模态分解EMD](https://zhuanlan.zhihu.com/p/40005057)
>
  [经验模态分解EMD IMF](https://zhuanlan.zhihu.com/p/44833026)
>
3 [改善EMD端点效应的方法](https://zhuanlan.zhihu.com/p/29050617)

# 1月18日
1 leetcode[permutation]
###
    class Solution(object):
        def permute(self, nums):
            """
            :type nums: List[int]
            :rtype: List[List[int]]
            """
            self.res = []
            sub = []
            self.dfs(nums,sub)
            return self.res

        def dfs(self, Nums, subList):
            if len(subList) == len(Nums):
                #print res,subList
                self.res.append(subList[:])
            for m in Nums:
                if m in subList:
                    continue
                subList.append(m)
                self.dfs(Nums,subList)
               subList.remove(m)

# 1月21日
1 工信部钢卷吞吐量预测

# 1月22日
1 [卡尔曼滤波器](https://www.zhihu.com/question/22422121)
>
2 [卡尔曼滤波器理解](https://www.jianshu.com/p/d3b1c3d307e0)
>
3 phm2010 刀具磨损值预测

# 1月23日
1 [红色石头机器学习、深度学习](https://redstonewill.com/)
>
2 [特征选择slelct k best](https://zhuanlan.zhihu.com/p/33199547)
>

# 1月25日
1 [简历制作tool](https://www.cakeresume.com/)
>

# 2月15日
1 [合页损失函数](https://blog.csdn.net/lz_peter/article/details/79614556)
2 [数据清洗常见方法](https://towardsdatascience.com/the-simple-yet-practical-data-cleaning-codes-ad27c4ce0a38)

# 2月16日
1 [深度学习吴恩达基础](http://kyonhuang.top/Andrew-Ng-Deep-Learning-notes/#/Convolutional_Neural_Networks/卷积神经网络)
2 [时间序列模型预处理](https://cloud.tencent.com/developer/article/1042809)

# 2月17日
1 [1st 特征工程demo](https://www.zhihu.com/question/21229371/answer/559468427)

# 2月21日
1 峨胜水泥 特征相关性分析

# 2月22日
1 [模型融合](https://zhuanlan.zhihu.com/p/40131797)
>
2 回归树基础(极客时间)
>
3 [随机森林](https://www.jianshu.com/p/a779f0686acc)
>
4 [梯度提升决策树](https://www.jianshu.com/p/005a4e6ac775)
>
5 模型总结从线性到非线性(极客时间)

# 2月25日
1 [自然语言处理 命名实体识别 图谱blog](https://blog.csdn.net/lhy2014/article/details/84582145#comments)

# 2月26日
1 [XGBOOST 参数](https://blog.csdn.net/zc02051126/article/details/46711047)
>
2 [机器学习算法中 GBDT 和 XGBOOST 的区别有哪些？](https://www.zhihu.com/question/41354392)

# 2月27日
1 友达光电异常设备检测(表的转化、iv算法)
>
2 [贝叶斯优化](https://blog.csdn.net/weixin_40105364/article/details/81322935)
>
3 [LABELIMG 快捷键](https://blog.csdn.net/qq_25254777/article/details/80352331)

# 3月1日
1 [深度学习相关blog 人体检测](https://blog.csdn.net/sinat_26917383/article/details/86557399)

# 3月4日
1 [信息量，信息熵，交叉熵，KL散度和互信息（信息增益）](https://blog.csdn.net/haolexiao/article/details/70142571)

# 3月11日
1 友达光电
>
2 wv,emlo(解决多意词),gpt,bert

# 3月12日
1 [卡方分布](https://blog.csdn.net/qq_39303465/article/details/79223843)
>
2 [卡方分布2](https://blog.csdn.net/snowdroptulip/article/details/78770088)

# 3月13日
1 [cramer's v]https://www.cnblogs.com/webRobot/p/6943562.html
>
2 [pandas表合并](https://www.cnblogs.com/bambipai/p/7668811.html)
>
3 卡方分布异常值检测
>

# 3月22日
1 [时间序列数据处理](https://blog.csdn.net/toshibahuai/article/details/79034829)

# 3月25日
1 [EM算法](https://blog.csdn.net/weixin_38206214/article/details/81431932)

# 3月26日
1 [贝叶斯估计](https://blog.csdn.net/zengxiantao1994/article/details/72889732)
>
2 [hmm前向计算的理解](https://www.zhihu.com/question/20962240?sort=created)

# 3月27日
1 [Python读取配置文件](https://www.cnblogs.com/cnhkzyy/p/9294829.html)
>
2 [hmm时间复杂度计算](https://www.cnblogs.com/liuwu265/p/4732797.html)

# 3月28日
1 [em算法](https://blog.csdn.net/yzheately/article/details/51164441)
>
2 [统计学习EM算法]
>

# 4月1日
1 [nlp知识图谱、关系抽取](https://blog.csdn.net/lhy2014)
>
2 [medical命名实体识别](https://blog.csdn.net/lhy2014/article/details/84582145)
>

# 4月3日
1 detectron平台使用：
>
1) 修改trainnet-thread 构造训练数据 dataset 放数据
2）修改模型配置文件num_class,IMS_PER_BATCH: 1,NMS: 0.5 MAX_ITER: 120000
  STEPS: [0, 40000, 80000]
>
2 [R-cnn]（https://www.cnblogs.com/zf-blog/p/6740736.html）
>

# 4月9日
1 [fast-rcnn详解](https://blog.csdn.net/u014380165/article/details/72851319)
>
2 [缺陷检测demo](https://blog.csdn.net/qq_27871973/article/details/86007150)

# 4月10日
1 [算法交易平台](http://ai.loliloli.pro/aicodes_home.html)

# 4月11日
1 [博客专家79名 nlp系列](https://blog.csdn.net/App_12062011/article/details/88370570)

# 4月15日
1 [python字典底层理解](https://blog.csdn.net/weixin_42681866/article/details/82785127)

# 4月16日
1 [找零问题](https://blog.csdn.net/sinat_30537123/article/details/78354349)

# 4月17日
1 
###
    class Solution:
        def numTrees(self, n):
            # The number of binary trees in i
            dp = [0 for _ in range(n+1)]
            # init
            dp[0], dp[1] = 1, 1
            for i in range(2, n+1):
                # j is the root 
                for j in range(1,i+1):
                    # the root of left subtree is -1,and the right is i-j
                    dp[i] += dp[j-1]*dp[i-j]
        
        return dp[n]

>
2 [安装docker](https://www.cnblogs.com/rookie404/p/5965518.html)
>
 [docker 权限不够](https://github.com/moby/moby/issues/2645)
>
 [docker申请账号](https://hub.docker.com/signup)

# 4月18日
1 docker run报错：
改为：nvidia-docker run -d -p 8888:8888 -p 6006:6006 -v /home/root/tensorflow-tensorlog:/tensorlog -v /home/root/tensorflow-data:/notebooks -v /var/tensorflow-dataset:/mnt -e PASSWORD=233233 docker.io/keineahnung2345/tensorflow-opencv:test2
>
2 [docker tensorflow](https://github.com/Ceruleanacg/Personae/issues/14）
>
3 [nvidia-docker安装](https://blog.csdn.net/A632189007/article/details/78801166)

# 4月19日
1 [yolo系列](https://zhuanlan.zhihu.com/p/46691043)
>
2 [yolo系列](https://www.jianshu.com/p/d13ae1055302)
>

# 4月24日
1 567. Permutation in String
###
    class Solution:
    """
    滑动窗口法，时间复杂度O(l1+26∗(l2−l1),l1是s1的长度,l2是s2的长度
    """
    def match(self, s1, s2):
        for i in range(26):
            if s1[i] != s2[i]:
                return False
        return True

    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1) > len(s2):
            return False
        list1 = [0 for i in range(26)]
        list2 = [0 for i in range(26)]
        for i in range(len(s1)):
            list1[ord(s1[i]) - ord('a')] += 1
            list2[ord(s2[i]) - ord('a')] += 1
        for i in range(len(s2) - len(s1)):
            if self.match(list1, list2):
                return True
            list2[ord(s2[i + len(s1)]) - ord('a')] += 1
            list2[ord(s2[i]) - ord('a')] -= 1
        return self.match(list1, list2)
 >
 2 230. Kth Smallest Element in a BST
 ###
    class Solution(object):
    def traverse(self, root, arr, k):
        if len(arr) == k:
            return arr
        if root.left != None:
            self.traverse(root.left, arr, k)
        arr.append(root.val)
        if root.right != None:
            self.traverse(root.right, arr, k)
        return arr 

    def kthSmallest(self, root, k):
        ret_list = self.traverse(root, [], k)        
        return ret_list[k-1] 
>

###
    def kthSmallest(self, root, k):
    self.k = k
    self.res = None
    self.helper(root)
    return self.res

    def helper(self, node):
        if not node:
            return
        self.helper(node.left)
        self.k -= 1
        if self.k == 0:
            self.res = node.val
            return
        self.helper(node.right)
 >
 ###
    def kthSmallest(root, k):
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right
 
# 4月25日
1 [maskrcnn benchmark 别人的docker](https://blog.csdn.net/u011420347/article/details/84782475)
>
2 [maskrcnn benchmark 自己的docker](https://zhuanlan.zhihu.com/p/55516749)

# 4月26日
1 [docker images删除](https://www.cnblogs.com/q4486233/p/6482711.html)
>
2 
### 55 jump Game
>
不管每一步怎么跳，我都跳到最后，跳到不能跳为止。 
比如我们用一个变量G，来记录我能跳到的最后的位置。对第i步来说，从第i个位置出发的最远是nums[i]+i那么我们的G=max(G,nums[i]+i) 
如果在某一步i>G，也就是说，前面能跳到的最远距离跳不到i，那就肯定失败。 
如果最终能跳到最后一步就返回成功。
>
    class Solution(object):
        def canJump(self, nums):
            """
            :type nums: List[int]
            :rtype: bool
            """
            L = len(nums)
            if L==0:
                return False
            G = nums[0]
            for i in range(1,L):
                if G<i:
                    return False
                G=max(G, nums[i]+i)
                if G>=L-1:
                    return True
            return True

### 4月28日
1 [nlp系列](https://blog.csdn.net/app_12062011/article/category/8102623)
>

### 4月30日
1 detectron 注意xml不要utf-8,不要中文保存路径。

# 5月5日
1 [超参数优化](https://roamanalytics.com/2016/09/15/optimizing-the-hyperparameter-of-which-hyperparameter-optimizer-to-use/#Packages)
>
2 [xgboost调参](https://www.cnblogs.com/mfryf/p/6293814.html)
>
3 [Randomsearch](https://cloud.tencent.com/developer/ask/124273)

# 5月6日
1 [透视表，交叉表](https://blog.csdn.net/bqw18744018044/article/details/80015840)
>
2 [python算法数据结构](https://github.com/TheAlgorithms/Python)
>
3 [docker安装tensorflow-gpu](https://tensorflow.google.cn/install/docker)

# 5月17日
1 ###
    class Solution:
        def coinChange(self,coins, amount):
            """
            :type coins: List[int]
            :type amount: int
            :rtype: int
            """


            if amount == 0:
                return 0
            Max = float('inf')
            result = [Max] * (amount + 1)
            result[0] = 0
            for i in range(1,amount + 1):
                for j in coins:
                    if i >= j:
                        result[i] = min(result[i],result[i-j] + 1)

        #         print('i',i)

            if result[amount] == Max:
                return -1
            else:
                return result[-1]
 
 # 5月18日
 1 [比赛代码https://github.com/geekinglcq/CDCS]
 >
 2 [nlphttps://github.com/fighting41love/funNLP]
 >
 3 [(注意力机制理解)https://zhuanlan.zhihu.com/p/37601161]
 >
 4 [(注意力机制理解2)https://www.jianshu.com/p/c94909b835d6]
 >
 5 [(机器学习面试博客)https://www.jianshu.com/u/511ba5d71aef]
 > 
 6 [(rnn与lstm)https://www.cnblogs.com/softzrp/p/6776398.html]
 >
 
# 5月21日
1 [基于互信息和左右信息熵的短语提取](https://www.jianshu.com/p/ceac583ec1f7)
>
2 [python3实现互信息和左右熵的新词发现]（https://www.jianshu.com/p/e9313fd692ef）
>

# 5月22日
1 [logistic 伯努利分布、指数分布理解](https://www.jianshu.com/p/a8d6b40da0cf)
>
2 [方差、样本方差、彻底理解样本方差为何除以n-1](https://blog.csdn.net/hearthougan/article/details/77859173)
>
3 [统计学习资料](https://www.cnblogs.com/think-and-do/p/6504272.html)
>
4 [反卷积、上采样](https://blog.csdn.net/nijiayan123/article/details/79416764)
>
5 [fpn](https://blog.csdn.net/u014380165/article/details/72890275/)

# 5月24日
1 [pytorch 基本网络架构](https://www.cnblogs.com/shenpings1314/p/10468418.html)
>
2 [行人识别](https://www.jiqizhixin.com/articles/2018-09-04-7)
>

# 5月27日
1 [fast-rcnn](https://blog.csdn.net/Katherine_hsr/article/details/80363191)
>
2 [fast-rcnn](https://www.jianshu.com/p/802bbce17f74)
>
3 [rcnn到faster-rcnn](https://www.cnblogs.com/skyfsm/p/6806246.html)
>
4 [python装饰器](https://www.liaoxuefeng.com/wiki/897692888725344/923030547069856)
>
5 [python装饰器更详细的解释](https://www.cnblogs.com/cicaday/p/python-decorator.html)

# 5月28日
1 [删除清华镜像源](https://blog.csdn.net/weixin_39278265/article/details/84782550)
>
2 [中科大镜像](https://www.jianshu.com/p/363eff5842d4)
>
3 [conda跟新gcc到4.9](https://blog.csdn.net/xiamentingtao/article/details/78376419)

# 5月29日
1   gcc 到4.9
    ### 
        [root@DS-VM-Node239 ~]# yum install centos-release-scl -y
        [root@DS-VM-Node239 ~]# yum install devtoolset-3-toolchain -y
        [root@DS-VM-Node239 ~]# scl enable devtoolset-3 bash
        [root@DS-VM-Node239 ~]# gcc --version   

# 5月30日
1 [python深度拷贝](https://www.cnblogs.com/hokky/p/8476698.html)
>
2 ### 列变行
          
          
          
                  #对犯罪类别:Category; 用LabelEncoder进行编号  
    leCrime = preprocessing.LabelEncoder()
    crime = leCrime.fit_transform(train.Category)#39种犯罪类型  
    #用get_dummies因子化星期几、街区、小时等特征  
    days=pd.get_dummies(train.DayOfWeek)
    district = pd.get_dummies(train.PdDistrict)
    hour = train.Dates.dt.hour
    hour = pd.get_dummies(hour)
    #组合特征  
    trainData = pd.concat([hour, days, district], axis = 1) #将特征进行横向组合  
    trainData['crime'] = crime#追加'crime'列  
    days = pd.get_dummies(test.DayOfWeek)
    district = pd.get_dummies(test.PdDistrict)
    hour = test.Dates.dt.hour
    hour = pd.get_dummies(hour)
    testData = pd.concat([hour, days, district], axis=1)
    trainData

# 5月31日
1 [返回数组中间的数]
###
    class Solution(object):
        def middleNode(self, head):
            """
            :type head: ListNode
            :rtype: ListNode
            """
            slow = fast = head
            while fast != None and fast.next != None:
                slow = slow.next
                fast = fast.next.next
            return slow
 >
 2 [剑指offer](https://www.cnblogs.com/yanmk/p/9130681.html)
 >

# 6月6日
1 [单因素方差分析](https://www.cnblogs.com/violetchan/p/10918525.html)
>

# 6月10日
1 
R-CNN 采用的是 IoU 的阈值，这个 threshold 取 0.3，如果一个区域与 Ground tureth 的 IoU 值低于设定的阈值，那么可以讲它看成是 Negetive.

IoU 的 threshold 它不是作者胡乱取值的，而是来自 {0,0.1,0.2,0.3,0.4,0.5} 的数值组合的。

而且，这个数值至关重要，如果 threshold 取值为 0.5,mAP 指标直接下降 5 个点，如果取值为 0，mAP 下降 4 个点。

一旦特征抽取成功，R-CNN 会用 SVM 去识别每个区域的类别，但这需要优化。

因为训练的数据太大，不可能一下子填充到电脑内存当中，R-CNN 作者采取了一种叫做 Hard negetive mining 的手段。

# 6月17日
1 [数据清洗缺失值](https://cloud.tencent.com/developer/article/1430000)
>
2 [特征工程]（https://www.jqr.com/article/000397）

>

# 6月18日
1 [课程视频制作](https://blog.csdn.net/ffffffff8/article/details/84033769)
>

# 6月24日
1 gcc更新（https://blog.csdn.net/u010103202/article/details/89060538）
>

# 6月25日
1 Python copy()
    copy dic key not value
###
        # -*- coding: utf-8 -*-
    """
    Created on Tue Jun 25 13:53:11 2019

    @author: root
    """
    import pandas as pd
    import numpy as np
    s = {}
    m = []
    s['a'] = 1
    s['b'] = 2
    v = s.copy()
    m.append(v)
    s['a'] = 3
    s['b'] = 4
    v = s.copy()
    m.append(v)

    s['a'] = 5
    s['b'] = 6
    v = s.copy()
    m.append(v)

    s['a'] = 7
    s['b'] = 8
    v = s.copy()
    m.append(v)
    m = pd.DataFrame(m)

# 6月26日
1 [解决测试冻结不了参数](! pip install -U --force-reinstall --no-dependencies git+https://github.com/datumbox/keras@fork/keras2.2.4)
>
2 [nvidia-docker 安装]
    wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb

    sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
>
3 [pytorch 各个模型fine-tune](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#comparison-with-model-trained-from-scratch)
>
4 [torchvision 模型代码](https://github.com/pytorch/vision/tree/master/torchvision)
>
5 [pytorch inception 两个输出 关掉一个](http://blog.sina.com.cn/s/blog_4b6a6ae50102yda3.html)

# 6月27日
1 [fc,maxpool,avgpool比较](https://www.cnblogs.com/hutao722/p/10008581.html)
>
2 maxpooling2d -> output [Npne,7,7,512]
> globalavgpooling -> output [None.512]
>
3 [resnext](https://zhuanlan.zhihu.com/p/32913695)
>

# 7月1日
1 [xgboost feature_importance](https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost)
>
2 [xgboost feature importance](https://www.cnblogs.com/haobang008/p/5929378.html)
>
3 [sppnet 从原图映射到featuremap](https://blog.csdn.net/liuxiaoheng1992/article/details/81775007)
>

# 7月3日
1 ###
        深度学习中经常看到epoch、 iteration和batchsize，下面按自己的理解说说这三个的区别：
    （1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
    （2）iteration：1个iteration等于使用batchsize个样本训练一次；
    （3）epoch：1个epoch等于使用训练集中的全部样本训练一次；
    举个例子，训练集有1000个样本，batchsize=10，那么：
    训练完整个样本集需要：
    100次iteration，1次epoch。
    关于batchsize可以看看这里

# 7月4日
1 ###
        import numpy as np
        a = None
        b = [[1,2,3]]
        print(b == None)

        b = np.array([[1,2,3]])
        print(b == None)

# 7月14日
1 [图片拼接]（https://blog.csdn.net/ahaotata/article/details/84027000）
>

# 7月17日
1 rpn:conv->slide-window->anchors

# 7月24日
1 [直观理解理解cnn]（https://www.jianshu.com/p/fc9175065d87）
>
2 [感受野计算](https://iphysresearch.github.io/posts/receptive_field.html)

# 7月29日
1 [安装TensorFlow 2.0](pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==2.0.0-beta1)
>

# 7月30日
1 [多标签分类](https://github.com/ItchyHiker/Multi_Label_Classification_Keras)
>
2 [培训博客](https://www.cbedai.net/u011630575/)

# 7月31日
1 [pytorch 交叉熵包含了softmax](https://blog.csdn.net/jiangpeng59/article/details/79583292)
>
2 [交叉熵损失函数理解](https://www.cnblogs.com/aijianiula/p/9460842.html)
>
3 [两种交叉熵损失函数区别](https://blog.csdn.net/u012436149/article/details/69660214)


# 8月8日
1 [可变分离卷积，以及1x1 conv 计算时将多通道信息融合](https://zhuanlan.zhihu.com/p/32746221)
>
2 [可变卷积](https://zhuanlan.zhihu.com/p/28749411)
