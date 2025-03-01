# 树形 DP - 直径系列

## [1.二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/description/)
思路：类似于求树的最大深度，求树的最大链长（最大深度-1）。遍历每个节点时，顺便计算经过当前节点的最大直径 = 左子树链长 + 右子树链长 + 2.   
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

## [2.二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/)
思路：每次返回当前子树的最大链和v，若v小于0则返回0。遍历每个节点时顺带更新答案ans=max(ans, node.val + leftv + rightv).  
代码：
```
class Solution {

    public int ans = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        dfs(root);
        return ans;
    }

    public int dfs(TreeNode node){
        if(node == null){
            return 0;
        }
        int leftVal = dfs(node.left);
        int rightVal = dfs(node.right);
        ans = Math.max(node.val + leftVal + rightVal, ans); //更新答案
        return Math.max(Math.max(node.val + leftVal, node.val + rightVal), 0);
    }
}
```

# 树形 DP - 最大独立集

## [1.打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/description/)
思路：  
![image](https://github.com/user-attachments/assets/0f432337-a910-4b4c-bf15-2c6d7a440a30)
  
代码：  
```
class Solution {
    public int rob(TreeNode root) {
        int[] ans = dfs(root);
        return Math.max(ans[0], ans[1]);
    }

    //当前节点有选和不选两种状态
    //返回两种状态(选，不选)下，当前树的最大值
    int[] dfs(TreeNode node){
        if(node == null)
            return new int[] {0, 0};
        int[] leftVal = dfs(node.left);
        int[] rightVal = dfs(node.right);
        int[] nodeVal = new int[2];
        nodeVal[0] = node.val + leftVal[1] + rightVal[1]; //选择当前节点
        nodeVal[1] = Math.max(leftVal[0], leftVal[1]) + 
                        Math.max(rightVal[0], rightVal[1]); //不选择当前节点
        return nodeVal;
    }
}
```


# 树形 DP - 最小支配集

## [1.监控二叉树](https://leetcode.cn/problems/binary-tree-cameras/description/)
思路：详见[灵神视频](https://www.bilibili.com/video/BV1oF411U7qL/?vd_source=1b49919c7c0227f4fcf55e85034bd317)或者代码注释.  

代码：  
```
class Solution {
    public int minCameraCover(TreeNode root) {
        int[] ans = dfs(root);
        return Math.min(ans[0], ans[2]);
    }

    //返回当前节点[蓝色，黄色，红色]状态下的最小摄像头次数
    //蓝色表示当前节点有摄像头
    //黄色表示父节点有摄像头
    //红色表示至少有一个子节点有摄像头
    public int[] dfs(TreeNode node){
        if(node == null)
            return new int[] {1001, 0, 0};
        int[] leftVal = dfs(node.left);
        int[] rightVal = dfs(node.right);
        int[] nodeVal = new int[3];
        //蓝色状态
        nodeVal[0] =  Math.min(Math.min(leftVal[0], leftVal[1]), leftVal[2])
         + Math.min(Math.min(rightVal[0], rightVal[1]), rightVal[2]) + 1;
        //黄色状态
        nodeVal[1] = Math.min(leftVal[0], leftVal[2]) + Math.min(rightVal[0], rightVal[2]);
        //红色状态
        nodeVal[2] = Math.min(Math.min(leftVal[0] + rightVal[0], leftVal[0] + rightVal[2]),
                                    leftVal[2] + rightVal[0]);
        return nodeVal;
    }
}
```
