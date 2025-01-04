# 区间 DP

## [1.最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/description/)
思路：dp[i][j]表示s[i]到s[j]之间的最长回文子序列长度。状态转移方程: 若s[i]=s[j], dp[i][j] = dp[i+1][j-1] + 2; 若s[i]!=s[j], dp[i][j] = max(dp[i+1][j], dp[i][j-1])。   
代码：
```
class Solution {
    public int longestPalindromeSubseq(String s) {
        char[] str = s.toCharArray();
        int n = str.length;
        int[][] dp = new int[n][n]; //dp[i][j]表示s[i]到s[j]之间的最长回文子序列长度。
        //状态转移方程
        //若s[i]=s[j], dp[i][j] = dp[i+1][j-1] + 2
        //若s[i]!=s[j], dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        //初始化，若i>j，dp[i][j]=0
        for(int i = n-1; i >= 0; i--){
            dp[i][i] = 1;
            for(int j = i+1; j < n; j++){
                if(str[i] == str[j]) dp[i][j] = dp[i+1][j-1] + 2;
                else dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
            }
        }
        return dp[0][n-1];
    }
}
```

## [2.多边形三角剖分的最低得分](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/description/)
思路：dp[i][j]表示顶点i顺时针到顶点j + 边j-i 围成的多边形最低分。状态转移方程: dp[i]][j] = min_(i < k < j){dp[i][k] + dp[k][j] + v[i]*v[k]*v[j]}。   
代码：
```
class Solution {
    public int minScoreTriangulation(int[] values) {
        int n = values.length;
        int[][] dp = new int[n][n]; //dp[i][j]表示顶点i顺时针到顶点j + 边j-i 围成的多边形最低分。
        //状态转移方程:
        //dp[i]][j] = min_(i < k < j){dp[i][k] + dp[k][j] + v[i]*v[k]*v[j]}
        for(int i = n-2; i >= 0; i--){ //i从倒数第二个点开始。
            dp[i][i+1] = 0; //只有一条边，低分为0
            for(int j = i+2; j < n; j++){
                dp[i][j] = Integer.MAX_VALUE;
                for(int k = i+1; k < j; k++){
                    dp[i][j] = 
                    Math.min(dp[i][k] + dp[k][j] + values[i]*values[k]*values[j], dp[i][j]);
                }
            }
        }
        return dp[0][n-1];
    }
}
```

## [3.相同分数的最大操作数目 II](https://leetcode.cn/problems/maximum-number-of-operations-with-the-same-score-ii/description/)
思路：dp[i][j]表示下标i到j的最多可操作次数。状态转移方程dp[i][j] = max(dp[i+2][j], dp[i][j-2], dp[i+1][j-1]) + 1 (各个操作需要满足删除的元素和为target)   
代码：
```
class Solution {
    public int maxOperations(int[] nums) {
        int n = nums.length;
        int res1 = dpResolve(nums, nums[0]+nums[1]);
        int res2 = dpResolve(nums, nums[n-2]+nums[n-1]);
        int res3 = dpResolve(nums, nums[0]+nums[n-1]);
        return Math.max(res1, Math.max(res2, res3));
    }

    public int dpResolve(int[] nums, int target){
        int n = nums.length;
        int[][] dp = new int[n][n]; //dp[i][j]表示下标i到j的最多可操作次数
        //状态转移方程dp[i][j] = max(dp[i+2][j], dp[i][j-2], dp[i+1][j-1]) + 1
        //(各个操作需要满足删除的元素和为target)
        for(int i = n-2; i >= 0; i--){
            if(nums[i] + nums[i+1] == target) dp[i][i+1] = 1;
            for(int j = i+2; j < n; j++){
                if(nums[i] + nums[i+1] == target) 
                    dp[i][j] = Math.max(dp[i+2][j] + 1, dp[i][j]);
                if(nums[j] + nums[j-1] == target) 
                    dp[i][j] = Math.max(dp[i][j-2] + 1, dp[i][j]);
                if(nums[i] + nums[j] == target) 
                    dp[i][j] = Math.max(dp[i+1][j-1] + 1, dp[i][j]);
            }
        }
        return dp[0][n-1];
    }
}
```

## [4.切棍子的最小成本](https://leetcode.cn/problems/minimum-cost-to-cut-a-stick/description/)
思路：注意这题的区间表示和前面题目略微有点不同。dp[i][j]表示切开索引i到j的切棍子最小总成本。状态转移方程 dp[i][j] = min_{k为中间切开索引}(dp[i][k] + dp[k][j]) + (cut[j] - cut[i])  
代码：
```
class Solution {
    public int minCost(int n, int[] cuts) {
        //预处理操作
        int[] new_cuts = new int[cuts.length+2];
        for(int i = 0; i < cuts.length; i++)
            new_cuts[i+1] = cuts[i];
        new_cuts[0] = 0;
        new_cuts[new_cuts.length-1] = n;
        Arrays.sort(new_cuts);
        int num = new_cuts.length; 
        int[][] dp = new int[num][num]; //dp[i][j]表示切开索引i到j的切棍子最小总成本
        //状态转移方程 dp[i][j] = min_{k为中间切开索引}(dp[i][k] + dp[k][j]) + (cut[j] - cut[i])
        for(int i = num-2; i >= 0; i--){
            for(int j = i+1; j < num; j++){
                for(int k = i+1; k < j; k++){ //查找k满足: i < k < j
                    int length = new_cuts[j] - new_cuts[i];
                    if(k == i+1) dp[i][j] = dp[i][k] + dp[k][j] + length;
                    else dp[i][j] = Math.min(dp[i][k] + dp[k][j] + length, dp[i][j]);
                }
            }
        }
        return dp[0][num-1];
    }
}
```
