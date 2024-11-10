# 代表性题目

## BFS模板
```
首元素入队
while(队列不为空)
{
  s=弹出队头元素。
  print（s）
  s的所有节点入队
}
```

## [1.二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/)
思路：当队列不为空时，队列的元素个数就是当前层的节点数，详见代码。     
代码：
```
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();

        List<TreeNode> queue = new LinkedList<>();
        if(root != null) queue.addLast(root);
        while(!queue.isEmpty())
        {
            List<Integer> tmp = new ArrayList<>();
            int n = queue.size(); //得到当前层的节点数
            for(int i = 0; i < n; i++) //遍历当前层的所有节点
            {
                TreeNode node = queue.removeFirst();
                tmp.add(node.val);
                if(node.left != null) queue.addLast(node.left);
                if(node.right != null) queue.addLast(node.right);
            }
            ans.add(tmp);
        }
        return ans;
    }
}
```

## [2.二叉树的锯齿形层序遍历]()
思路：和上一题类似，然后偶数层需要反转。   
代码：
```
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        List<TreeNode> queue = new LinkedList<>();
        if(root != null) queue.addLast(root);
        int depth = 0; //偶数层反转
        while(!queue.isEmpty())
        {
            List<Integer> temp = new ArrayList<>();
            int n = queue.size();
            depth++;
            for(int i = 0; i < n; i++)
            {
                TreeNode node = queue.removeFirst();
                temp.add(node.val);
                if(node.left != null) queue.addLast(node.left);
                if(node.right != null) queue.addLast(node.right);
            }
            if(depth % 2 == 0)
                Collections.reverse(temp);
            ans.add(temp);
        }   
        return ans;              
    }
}
```
