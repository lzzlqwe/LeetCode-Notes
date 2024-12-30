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
