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
