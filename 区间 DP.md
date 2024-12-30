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
