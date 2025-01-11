# 树形 DP - 直径系列

## [1.二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/description/)
思路：类似于求树的最大深度，求数的最大链长（最大深度-1）。遍历每个节点时，顺便计算经过当前节点的最大直径 = 左子树链长 + 右子树链长 + 2.   
代码：
```
class Solution {

    public int ans = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return ans;
    }

    public int dfs(TreeNode node){
        if(node == null)
            return -1;
        int left = dfs(node.left);
        int right = dfs(node.right);
        int diameter = left + right + 2; //求经过当前节点的直径
        ans = Math.max(diameter, ans); //更新全局最大直径
        return Math.max(left, right) + 1; //返回当前最大链长
    }
}
```
